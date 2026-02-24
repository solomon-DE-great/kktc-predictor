"""
KKTC Arrest Prediction System — Database Layer
SQLite with spatial extensions for MVP, upgrade to PostGIS for production.
"""
import sqlite3
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from config import DB_PATH, DATA_DIR


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS arrests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,              -- YYYY-MM-DD
            time TEXT,                       -- HH:MM
            district TEXT NOT NULL,
            specific_area TEXT,
            count INTEGER DEFAULT 1,
            reason TEXT DEFAULT 'ikamet izinsiz',
            day_of_week TEXT,
            hour_decimal REAL,
            latitude REAL,
            longitude REAL,
            operating_unit TEXT,
            source_url TEXT,
            source_name TEXT,
            raw_text TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            -- Derived features (computed on insert)
            is_weekend INTEGER DEFAULT 0,
            month INTEGER,
            season TEXT,
            time_window TEXT,
            is_industrial_zone INTEGER DEFAULT 0,
            UNIQUE(date, time, district, specific_area, count)
        );

        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            source TEXT NOT NULL,
            title TEXT,
            body TEXT,
            publish_date TEXT,
            scraped_at TEXT DEFAULT (datetime('now')),
            processed INTEGER DEFAULT 0,
            relevant INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_at TEXT DEFAULT (datetime('now')),
            target_date TEXT NOT NULL,
            district TEXT NOT NULL,
            specific_area TEXT,
            predicted_time_window TEXT,
            predicted_hour REAL,
            predicted_count REAL,
            confidence REAL,
            latitude REAL,
            longitude REAL,
            model_version TEXT,
            was_correct INTEGER DEFAULT NULL  -- Fill in after the fact
        );

        CREATE TABLE IF NOT EXISTS scrape_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            scraped_at TEXT DEFAULT (datetime('now')),
            articles_found INTEGER DEFAULT 0,
            new_articles INTEGER DEFAULT 0,
            arrests_extracted INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_arrests_date ON arrests(date);
        CREATE INDEX IF NOT EXISTS idx_arrests_district ON arrests(district);
        CREATE INDEX IF NOT EXISTS idx_arrests_coords ON arrests(latitude, longitude);
        CREATE INDEX IF NOT EXISTS idx_articles_processed ON articles(processed);
        CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(target_date);
    """)
    conn.commit()
    conn.close()
    print(f"[OK] Database initialized at {DB_PATH}")


def insert_arrest(data: dict) -> bool:
    """Insert a single arrest record. Returns True if new, False if duplicate."""
    conn = get_connection()

    # Compute derived features
    if data.get("date"):
        dt = datetime.strptime(data["date"], "%Y-%m-%d")
        data["day_of_week"] = dt.strftime("%A")
        data["is_weekend"] = 1 if dt.weekday() >= 5 else 0
        data["month"] = dt.month
        months_to_season = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn",
        }
        data["season"] = months_to_season.get(dt.month, "unknown")

    if data.get("time"):
        parts = data["time"].split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        data["hour_decimal"] = h + m / 60.0

        if 5 <= h < 8:
            data["time_window"] = "early_morning"
        elif 8 <= h < 12:
            data["time_window"] = "morning"
        elif 12 <= h < 17:
            data["time_window"] = "afternoon"
        elif 17 <= h < 21:
            data["time_window"] = "evening"
        else:
            data["time_window"] = "night"

    area = (data.get("specific_area") or "").lower()
    data["is_industrial_zone"] = 1 if any(
        kw in area for kw in ["sanayi", "industrial", "organize"]
    ) else 0

    try:
        conn.execute("""
            INSERT INTO arrests (
                date, time, district, specific_area, count, reason,
                day_of_week, hour_decimal, latitude, longitude,
                operating_unit, source_url, source_name, raw_text,
                is_weekend, month, season, time_window, is_industrial_zone
            ) VALUES (
                :date, :time, :district, :specific_area, :count, :reason,
                :day_of_week, :hour_decimal, :latitude, :longitude,
                :operating_unit, :source_url, :source_name, :raw_text,
                :is_weekend, :month, :season, :time_window, :is_industrial_zone
            )
        """, {
            "date": data.get("date"),
            "time": data.get("time"),
            "district": data.get("district"),
            "specific_area": data.get("specific_area"),
            "count": data.get("count", 1),
            "reason": data.get("reason", "ikamet izinsiz"),
            "day_of_week": data.get("day_of_week"),
            "hour_decimal": data.get("hour_decimal"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "operating_unit": data.get("operating_unit"),
            "source_url": data.get("source_url"),
            "source_name": data.get("source_name"),
            "raw_text": data.get("raw_text"),
            "is_weekend": data.get("is_weekend", 0),
            "month": data.get("month"),
            "season": data.get("season"),
            "time_window": data.get("time_window"),
            "is_industrial_zone": data.get("is_industrial_zone", 0),
        })
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def insert_prediction(pred: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO predictions (
            target_date, district, specific_area, predicted_time_window,
            predicted_hour, predicted_count, confidence, latitude, longitude,
            model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pred["target_date"], pred["district"], pred.get("specific_area"),
        pred.get("predicted_time_window"), pred.get("predicted_hour"),
        pred.get("predicted_count"), pred["confidence"],
        pred.get("latitude"), pred.get("longitude"),
        pred.get("model_version", "v1"),
    ))
    conn.commit()
    conn.close()


def get_all_arrests() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM arrests ORDER BY date DESC, time DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_arrests_since(days_ago: int = 90) -> list[dict]:
    cutoff = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM arrests WHERE date >= ? ORDER BY date, time", (cutoff,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_arrest_count():
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM arrests").fetchone()[0]
    conn.close()
    return count


def get_latest_predictions(n: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM predictions
        ORDER BY predicted_at DESC, confidence DESC
        LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def import_csv(csv_path: str) -> int:
    """Import existing CSV data into the database. Returns count of new records."""
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = {
                "date": row.get("Date", row.get("date")),
                "time": row.get("Time", row.get("time")),
                "district": row.get("District", row.get("district")),
                "specific_area": row.get("Specific Area/Street", row.get("specific_area")),
                "count": int(row.get("Count", row.get("count", 1))),
                "reason": row.get("Reason", row.get("reason", "ikamet izinsiz")),
                "latitude": float(row.get("Latitude", row.get("latitude", 0))) or None,
                "longitude": float(row.get("Longitude", row.get("longitude", 0))) or None,
                "source_name": "csv_import",
            }
            if insert_arrest(data):
                count += 1
    return count


def export_to_json(path: str = None) -> str:
    """Export all arrests to JSON for the dashboard."""
    arrests = get_all_arrests()
    predictions = get_latest_predictions(50)
    output = {
        "arrests": arrests,
        "predictions": predictions,
        "stats": {
            "total_records": len(arrests),
            "total_people": sum(a.get("count", 1) for a in arrests),
            "districts": len(set(a["district"] for a in arrests)),
            "date_range": {
                "earliest": arrests[-1]["date"] if arrests else None,
                "latest": arrests[0]["date"] if arrests else None,
            },
        },
        "exported_at": datetime.now().isoformat(),
    }
    path = path or str(DATA_DIR / "dashboard_data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    init_db()
    print(f"Total arrests in database: {get_arrest_count()}")
