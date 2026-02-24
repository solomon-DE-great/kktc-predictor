"""
KKTC Arrest Prediction System — Prediction Engine
Multi-model approach combining:
  1. Frequency analysis (baseline)
  2. Time-series forecasting (Prophet)
  3. Spatial kernel density estimation (KDE)
  4. Classification model (XGBoost/RandomForest)
  5. Temporal pattern matching
"""
import json
import logging
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from math import exp, sqrt, pi, radians, cos, sin, asin

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    DISTRICTS, TIME_WINDOWS, KNOWN_LOCATIONS,
    MIN_RECORDS_FOR_PREDICTION, PREDICTION_HORIZON_DAYS, DATA_DIR
)
from database import get_all_arrests, get_arrests_since, insert_prediction, get_arrest_count

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    """Distance between two points in km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))


def time_window_for_hour(hour: float) -> str:
    if 5 <= hour < 8: return "early_morning"
    if 8 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 21: return "evening"
    return "night"


def day_number(day_name: str) -> int:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    try:
        return days.index(day_name)
    except ValueError:
        return 0


# ═══════════════════════════════════════════════════════════
# MODEL 1: FREQUENCY ANALYSIS (works with ANY amount of data)
# ═══════════════════════════════════════════════════════════

class FrequencyModel:
    """Baseline model using simple frequency distributions."""

    def __init__(self, arrests: list[dict]):
        self.arrests = arrests
        self._analyze()

    def _analyze(self):
        # District frequency (weighted by count of people)
        self.district_freq = Counter()
        self.area_freq = Counter()
        self.day_freq = Counter()
        self.hour_buckets = Counter()
        self.area_details = defaultdict(lambda: {
            "count": 0, "total_people": 0, "hours": [], "days": [],
            "lat": None, "lng": None, "dates": []
        })

        for a in self.arrests:
            weight = a.get("count", 1)
            district = a.get("district", "Unknown")
            area = a.get("specific_area", "Unknown")
            key = f"{district}|{area}"

            self.district_freq[district] += weight
            self.area_freq[key] += 1
            self.day_freq[a.get("day_of_week", "Unknown")] += 1

            hour = a.get("hour_decimal")
            if hour is not None:
                self.hour_buckets[time_window_for_hour(hour)] += 1

            details = self.area_details[key]
            details["count"] += 1
            details["total_people"] += weight
            details["district"] = district
            details["area"] = area
            if hour is not None:
                details["hours"].append(hour)
            details["days"].append(a.get("day_of_week"))
            details["dates"].append(a.get("date"))
            if a.get("latitude"):
                details["lat"] = a["latitude"]
                details["lng"] = a["longitude"]

    def predict_hotspots(self, n: int = 10) -> list[dict]:
        """Top N most likely locations for next arrest."""
        total = sum(self.area_freq.values())
        predictions = []

        for key, count in self.area_freq.most_common(n):
            details = self.area_details[key]
            avg_hour = np.mean(details["hours"]) if details["hours"] else 12.0
            mode_day = Counter(details["days"]).most_common(1)[0][0] if details["days"] else "Wednesday"
            avg_count = round(details["total_people"] / details["count"])

            # Confidence based on frequency + recency
            freq_score = count / total
            recency_score = 0
            if details["dates"]:
                most_recent = max(details["dates"])
                try:
                    days_ago = (datetime.now() - datetime.strptime(most_recent, "%Y-%m-%d")).days
                    recency_score = max(0, 1 - days_ago / 90)
                except ValueError:
                    pass
            confidence = min(0.95, freq_score * 0.5 + recency_score * 0.3 + 0.1)

            predictions.append({
                "district": details["district"],
                "specific_area": details["area"],
                "predicted_hour": round(avg_hour, 1),
                "predicted_time_window": time_window_for_hour(avg_hour),
                "predicted_day": mode_day,
                "predicted_count": avg_count,
                "confidence": round(confidence, 3),
                "latitude": details["lat"],
                "longitude": details["lng"],
                "historical_events": count,
                "model": "frequency",
            })

        return predictions

    def predict_timing(self) -> dict:
        """Most likely day and time for next arrest."""
        return {
            "top_days": self.day_freq.most_common(),
            "top_time_windows": self.hour_buckets.most_common(),
            "total_events": len(self.arrests),
        }


# ═══════════════════════════════════════════════════════════
# MODEL 2: TEMPORAL PATTERN MATCHING
# ═══════════════════════════════════════════════════════════

class TemporalPatternModel:
    """Detects repeating patterns: same area raided on consecutive days,
    weekly cycles, bi-weekly operations, etc."""

    def __init__(self, arrests: list[dict]):
        self.df = pd.DataFrame(arrests)
        if not self.df.empty and "date" in self.df.columns:
            self.df["date_dt"] = pd.to_datetime(self.df["date"], errors="coerce")
            self.df = self.df.dropna(subset=["date_dt"])

    def find_repeat_patterns(self) -> list[dict]:
        """Find locations that get raided repeatedly and predict next raid."""
        if self.df.empty:
            return []

        patterns = []
        grouped = self.df.groupby(["district", "specific_area"])

        for (district, area), group in grouped:
            if len(group) < 2:
                continue

            dates = sorted(group["date_dt"].unique())
            if len(dates) < 2:
                continue

            # Calculate inter-arrest intervals
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals) if len(intervals) > 1 else avg_interval

            # Predict next date
            last_date = max(dates)
            predicted_next = last_date + pd.Timedelta(days=max(1, int(avg_interval)))

            # Confidence based on consistency of intervals
            if std_interval == 0:
                consistency = 0.9  # Perfect regularity
            else:
                cv = std_interval / max(avg_interval, 1)  # Coefficient of variation
                consistency = max(0.2, 1 - cv)

            # Time pattern
            hours = group["hour_decimal"].dropna()
            avg_hour = hours.mean() if not hours.empty else 12.0
            hour_std = hours.std() if len(hours) > 1 else 4.0

            patterns.append({
                "district": district,
                "specific_area": area,
                "event_count": len(group),
                "intervals_days": intervals,
                "avg_interval_days": round(avg_interval, 1),
                "predicted_next_date": predicted_next.strftime("%Y-%m-%d"),
                "predicted_hour": round(avg_hour, 1),
                "hour_uncertainty": round(hour_std, 1),
                "consistency_score": round(consistency, 3),
                "last_raid_date": last_date.strftime("%Y-%m-%d"),
                "latitude": group["latitude"].iloc[0] if "latitude" in group.columns else None,
                "longitude": group["longitude"].iloc[0] if "longitude" in group.columns else None,
                "model": "temporal_pattern",
            })

        return sorted(patterns, key=lambda x: x["consistency_score"], reverse=True)

    def detect_operation_days(self) -> dict:
        """Detect if police run multi-area operations on same day."""
        if self.df.empty:
            return {}

        daily_counts = self.df.groupby("date").agg(
            events=("district", "count"),
            districts=("district", "nunique"),
            total_people=("count", "sum")
        ).reset_index()

        # "Operation days" = days with 3+ events or 2+ districts
        ops = daily_counts[
            (daily_counts["events"] >= 3) | (daily_counts["districts"] >= 2)
        ]

        if ops.empty:
            return {"operation_days": [], "avg_events_per_op": 0}

        # What day of week do operations happen?
        ops["date_dt"] = pd.to_datetime(ops["date"])
        ops["dow"] = ops["date_dt"].dt.day_name()
        dow_counts = ops["dow"].value_counts().to_dict()

        return {
            "operation_days": ops.to_dict("records"),
            "preferred_days": dow_counts,
            "avg_events_per_op": round(ops["events"].mean(), 1),
            "avg_people_per_op": round(ops["total_people"].mean(), 1),
        }


# ═══════════════════════════════════════════════════════════
# MODEL 3: SPATIAL KERNEL DENSITY ESTIMATION
# ═══════════════════════════════════════════════════════════

class SpatialKDEModel:
    """Generates probability heatmaps showing where arrests are most likely."""

    def __init__(self, arrests: list[dict]):
        coords = [
            (a["latitude"], a["longitude"], a.get("count", 1))
            for a in arrests
            if a.get("latitude") and a.get("longitude")
        ]
        self.lats = np.array([c[0] for c in coords])
        self.lngs = np.array([c[1] for c in coords])
        self.weights = np.array([c[2] for c in coords])
        self.has_data = len(coords) >= 3

    def compute_heatmap(self, grid_resolution: int = 50) -> dict:
        """Generate a heatmap grid of arrest probability."""
        if not self.has_data:
            return {"grid": [], "bounds": {}}

        lat_min, lat_max = self.lats.min() - 0.02, self.lats.max() + 0.02
        lng_min, lng_max = self.lngs.min() - 0.02, self.lngs.max() + 0.02

        lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
        lng_grid = np.linspace(lng_min, lng_max, grid_resolution)

        try:
            # Use scipy KDE with weights
            positions = np.vstack([
                np.repeat(self.lats, self.weights.astype(int)),
                np.repeat(self.lngs, self.weights.astype(int))
            ])
            kernel = stats.gaussian_kde(positions, bw_method=0.15)

            xx, yy = np.meshgrid(lat_grid, lng_grid)
            grid_positions = np.vstack([xx.ravel(), yy.ravel()])
            density = kernel(grid_positions).reshape(xx.shape)

            # Normalize to 0-1
            density = (density - density.min()) / (density.max() - density.min() + 1e-10)

        except Exception as e:
            log.warning(f"KDE failed: {e}, using simple distance-based fallback")
            density = self._simple_heatmap(lat_grid, lng_grid)

        return {
            "grid": density.tolist(),
            "bounds": {
                "lat_min": lat_min, "lat_max": lat_max,
                "lng_min": lng_min, "lng_max": lng_max,
            },
            "lat_ticks": lat_grid.tolist(),
            "lng_ticks": lng_grid.tolist(),
            "hotspots": self._find_peaks(density, lat_grid, lng_grid),
        }

    def _simple_heatmap(self, lat_grid, lng_grid):
        """Fallback heatmap using weighted distance."""
        xx, yy = np.meshgrid(lat_grid, lng_grid)
        density = np.zeros_like(xx)
        bandwidth = 0.01

        for lat, lng, w in zip(self.lats, self.lngs, self.weights):
            dist_sq = (xx - lat)**2 + (yy - lng)**2
            density += w * np.exp(-dist_sq / (2 * bandwidth**2))

        if density.max() > 0:
            density /= density.max()
        return density

    def _find_peaks(self, density, lat_grid, lng_grid, threshold: float = 0.5) -> list[dict]:
        """Find peak density locations."""
        peaks = []
        for i in range(1, density.shape[0]-1):
            for j in range(1, density.shape[1]-1):
                val = density[i, j]
                if val < threshold:
                    continue
                # Check if local maximum
                neighbors = density[i-1:i+2, j-1:j+2]
                if val >= neighbors.max():
                    peaks.append({
                        "latitude": round(lat_grid[j], 5),
                        "longitude": round(lng_grid[i], 5),
                        "density": round(float(val), 4),
                    })
        return sorted(peaks, key=lambda x: x["density"], reverse=True)[:10]


# ═══════════════════════════════════════════════════════════
# MODEL 4: CLASSIFICATION (when enough data)
# ═══════════════════════════════════════════════════════════

class ClassificationModel:
    """Random Forest / Gradient Boosting to predict arrest risk for
    a given (district, day_of_week, time_window) combination."""

    def __init__(self, arrests: list[dict]):
        self.df = pd.DataFrame(arrests)
        self.model = None
        self.feature_names = []
        self.trained = False

        if len(self.df) >= MIN_RECORDS_FOR_PREDICTION:
            self._train()

    def _train(self):
        try:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import cross_val_score

            df = self.df.copy()

            # Create features
            le_district = LabelEncoder()
            df["district_enc"] = le_district.fit_transform(df["district"].fillna("Unknown"))
            self.le_district = le_district

            le_day = LabelEncoder()
            df["day_enc"] = le_day.fit_transform(df["day_of_week"].fillna("Unknown"))
            self.le_day = le_day

            features = ["district_enc", "day_enc", "hour_decimal", "is_weekend",
                        "month", "is_industrial_zone"]

            # Filter to features that exist
            features = [f for f in features if f in df.columns]
            self.feature_names = features

            X = df[features].fillna(0).values
            y = (df["count"].fillna(1) > 0).astype(int).values  # Binary: arrest happened

            # Try gradient boosting first
            try:
                self.model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
                )
                scores = cross_val_score(self.model, X, y, cv=min(3, len(X)), scoring="accuracy")
                log.info(f"GBM CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
            except Exception:
                self.model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)

            self.model.fit(X, y)
            self.trained = True
            log.info(f"Classification model trained on {len(X)} samples")

        except ImportError:
            log.warning("scikit-learn not installed. Install with: pip install scikit-learn")
        except Exception as e:
            log.error(f"Training failed: {e}")

    def predict_risk(self, district: str, day: str, hour: float) -> float:
        """Predict arrest probability for given conditions. Returns 0-1."""
        if not self.trained:
            return 0.5  # No model, return neutral

        try:
            district_enc = (
                self.le_district.transform([district])[0]
                if district in self.le_district.classes_
                else 0
            )
            day_enc = (
                self.le_day.transform([day])[0]
                if day in self.le_day.classes_
                else 0
            )
            is_weekend = 1 if day in ("Saturday", "Sunday") else 0
            month = datetime.now().month
            is_industrial = 0

            features = {
                "district_enc": district_enc,
                "day_enc": day_enc,
                "hour_decimal": hour,
                "is_weekend": is_weekend,
                "month": month,
                "is_industrial_zone": is_industrial,
            }
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            prob = self.model.predict_proba(X)[0][1]
            return round(float(prob), 3)
        except Exception as e:
            log.warning(f"Prediction error: {e}")
            return 0.5

    def get_feature_importance(self) -> dict:
        """Return feature importance rankings."""
        if not self.trained or not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_.round(4).tolist()))


# ═══════════════════════════════════════════════════════════
# MODEL 5: TIME-SERIES FORECASTING
# ═══════════════════════════════════════════════════════════

class TimeSeriesModel:
    """Daily arrest count forecasting using Prophet or simple methods."""

    def __init__(self, arrests: list[dict]):
        self.df = pd.DataFrame(arrests)
        self.forecast = None

    def predict_daily_counts(self, horizon_days: int = 7) -> list[dict]:
        """Forecast daily arrest counts for next N days."""
        if self.df.empty or "date" not in self.df.columns:
            return []

        # Aggregate daily counts
        daily = self.df.groupby("date")["count"].sum().reset_index()
        daily.columns = ["ds", "y"]
        daily["ds"] = pd.to_datetime(daily["ds"], errors="coerce")
        daily = daily.dropna().sort_values("ds")

        if len(daily) < 5:
            return self._simple_forecast(daily, horizon_days)

        try:
            from prophet import Prophet

            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False if len(daily) < 60 else True,
                changepoint_prior_scale=0.1,
            )
            m.fit(daily)

            future = m.make_future_dataframe(periods=horizon_days)
            forecast = m.predict(future)

            results = []
            for _, row in forecast.tail(horizon_days).iterrows():
                results.append({
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "day_of_week": row["ds"].strftime("%A"),
                    "predicted_count": max(0, round(row["yhat"], 1)),
                    "lower_bound": max(0, round(row["yhat_lower"], 1)),
                    "upper_bound": max(0, round(row["yhat_upper"], 1)),
                    "model": "prophet",
                })
            self.forecast = results
            return results

        except ImportError:
            log.info("Prophet not installed, using simple forecast")
            return self._simple_forecast(daily, horizon_days)

    def _simple_forecast(self, daily: pd.DataFrame, horizon: int) -> list[dict]:
        """Fallback: day-of-week average forecast."""
        if daily.empty:
            return []

        daily["dow"] = daily["ds"].dt.day_name()
        dow_avg = daily.groupby("dow")["y"].mean().to_dict()
        overall_avg = daily["y"].mean()

        results = []
        for i in range(1, horizon + 1):
            d = datetime.now() + timedelta(days=i)
            dow = d.strftime("%A")
            pred = dow_avg.get(dow, overall_avg)
            results.append({
                "date": d.strftime("%Y-%m-%d"),
                "day_of_week": dow,
                "predicted_count": round(pred, 1),
                "lower_bound": round(max(0, pred * 0.5), 1),
                "upper_bound": round(pred * 1.5, 1),
                "model": "day_of_week_avg",
            })
        self.forecast = results
        return results


# ═══════════════════════════════════════════════════════════
# MASTER PREDICTOR — Combines all models
# ═══════════════════════════════════════════════════════════

class MasterPredictor:
    """Orchestrates all models and produces unified predictions."""

    def __init__(self):
        self.arrests = get_all_arrests()
        self.n_records = len(self.arrests)
        log.info(f"Loaded {self.n_records} arrest records")

        # Initialize all models
        self.freq_model = FrequencyModel(self.arrests)
        self.temporal_model = TemporalPatternModel(self.arrests)
        self.kde_model = SpatialKDEModel(self.arrests)
        self.class_model = ClassificationModel(self.arrests)
        self.ts_model = TimeSeriesModel(self.arrests)

    def generate_full_prediction(self, horizon_days: int = 7, top_n: int = 10) -> dict:
        """Generate comprehensive predictions combining all models."""
        log.info(f"Generating predictions for next {horizon_days} days...")

        # 1. Frequency-based hotspots
        freq_predictions = self.freq_model.predict_hotspots(top_n)
        timing = self.freq_model.predict_timing()

        # 2. Temporal patterns (repeat raids)
        repeat_patterns = self.temporal_model.find_repeat_patterns()
        operation_patterns = self.temporal_model.detect_operation_days()

        # 3. Spatial heatmap
        heatmap = self.kde_model.compute_heatmap()

        # 4. Classification risk scores
        if self.class_model.trained:
            # Score each possible (district, day, time) combination
            for pred in freq_predictions:
                for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                    risk = self.class_model.predict_risk(
                        pred["district"], day, pred.get("predicted_hour", 12)
                    )
                    if risk > 0.6:
                        pred["ml_risk_score"] = risk
            feature_importance = self.class_model.get_feature_importance()
        else:
            feature_importance = {}

        # 5. Time series forecast
        daily_forecast = self.ts_model.predict_daily_counts(horizon_days)

        # 6. Combine and rank — UNIFIED PREDICTIONS
        unified = self._combine_predictions(
            freq_predictions, repeat_patterns, daily_forecast, horizon_days
        )

        # Store predictions
        for pred in unified[:20]:
            insert_prediction({
                "target_date": pred.get("predicted_date", ""),
                "district": pred["district"],
                "specific_area": pred.get("specific_area"),
                "predicted_time_window": pred.get("predicted_time_window"),
                "predicted_hour": pred.get("predicted_hour"),
                "predicted_count": pred.get("predicted_count"),
                "confidence": pred["confidence"],
                "latitude": pred.get("latitude"),
                "longitude": pred.get("longitude"),
                "model_version": "v1_combined",
            })

        result = {
            "generated_at": datetime.now().isoformat(),
            "data_stats": {
                "total_records": self.n_records,
                "model_confidence_note": self._confidence_note(),
            },
            "predictions": unified[:top_n],
            "daily_forecast": daily_forecast,
            "repeat_patterns": repeat_patterns[:10],
            "operation_patterns": operation_patterns,
            "heatmap": heatmap,
            "timing_analysis": timing,
            "feature_importance": feature_importance,
        }

        # Save to file
        output_path = str(DATA_DIR / "predictions.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"Predictions saved to {output_path}")

        return result

    def _combine_predictions(self, freq, patterns, daily, horizon) -> list[dict]:
        """Merge frequency hotspots with temporal patterns into date-specific predictions."""
        combined = []
        today = datetime.now()

        for pred in freq:
            # Find matching temporal pattern
            matching_pattern = None
            for pat in patterns:
                if pat["district"] == pred["district"] and pat.get("specific_area") == pred.get("specific_area"):
                    matching_pattern = pat
                    break

            # Generate date-specific prediction
            if matching_pattern:
                predicted_date = matching_pattern.get("predicted_next_date")
                consistency = matching_pattern.get("consistency_score", 0.3)
                final_confidence = min(0.95,
                    pred["confidence"] * 0.4 +
                    consistency * 0.4 +
                    0.2 * (matching_pattern["event_count"] / max(self.n_records, 1))
                )
            else:
                # No temporal pattern — use day-of-week heuristic
                target_day = pred.get("predicted_day", "Wednesday")
                days_ahead = (day_number(target_day) - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                predicted_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                final_confidence = pred["confidence"] * 0.7

            combined.append({
                **pred,
                "predicted_date": predicted_date,
                "confidence": round(final_confidence, 3),
                "has_temporal_pattern": matching_pattern is not None,
                "pattern_interval_days": matching_pattern["avg_interval_days"] if matching_pattern else None,
            })

        # Sort by confidence
        combined.sort(key=lambda x: x["confidence"], reverse=True)
        return combined

    def _confidence_note(self) -> str:
        if self.n_records < 30:
            return f"LOW: Only {self.n_records} records. Need 200+ for reliable predictions. Current output is pattern-suggestive only."
        elif self.n_records < 100:
            return f"MODERATE: {self.n_records} records provide basic patterns but predictions have wide uncertainty."
        elif self.n_records < 500:
            return f"GOOD: {self.n_records} records enable meaningful district-level predictions."
        else:
            return f"HIGH: {self.n_records} records provide strong statistical foundation for predictions."

    def quick_summary(self) -> str:
        """Human-readable summary of top predictions."""
        result = self.generate_full_prediction()
        lines = [
            f"=== KKTC ARREST PREDICTIONS ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===",
            f"Based on {self.n_records} historical records",
            f"Confidence level: {result['data_stats']['model_confidence_note']}",
            "",
            "TOP PREDICTED NEXT ARRESTS:",
        ]

        for i, p in enumerate(result["predictions"][:5], 1):
            lines.append(
                f"  {i}. {p['district']}/{p.get('specific_area', '?')} "
                f"— {p.get('predicted_date', '?')} ~{p.get('predicted_hour', '?')}:00 "
                f"(~{p.get('predicted_count', '?')} people, {p['confidence']*100:.0f}% conf)"
            )

        lines.append("\nDAILY FORECAST:")
        for d in result.get("daily_forecast", [])[:7]:
            lines.append(
                f"  {d['date']} ({d['day_of_week'][:3]}): "
                f"~{d['predicted_count']} arrests [{d['lower_bound']}-{d['upper_bound']}]"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    predictor = MasterPredictor()
    print(predictor.quick_summary())
