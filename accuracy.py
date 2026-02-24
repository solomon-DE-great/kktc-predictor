"""
KKTC Prediction Accuracy Tracker

Compares past predictions against actual arrest records and scores them.

Usage:
    python accuracy.py                               # evaluate last 7 days + weekly summary
    python accuracy.py check --from 2026-02-20 --to 2026-02-27
    python accuracy.py check --from 2026-02-20 --to 2026-02-27 --force
    python accuracy.py mark --prediction-id 3 --correct yes
    python accuracy.py mark --prediction-id 3 --correct no

Matching rules (a prediction is a HIT if ALL of these hold):
    1. Same district (case-insensitive)
    2. An arrest occurred within ±2 calendar days of the predicted date (≤ 48 h)
    3. If predicted_hour is known: arrest hour within 4 h of predicted hour
       Else if time_window is known: arrest time_window matches
       Else: district + date match alone is sufficient
"""

import argparse
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Unicode-safe output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = Path(__file__).parent / "data" / "arrests.db"

TW_LABELS = {
    "early_morning": "EARLY MORNING 05–08",
    "morning":       "MORNING 08–12",
    "afternoon":     "AFTERNOON 12–17",
    "evening":       "EVENING 17–21",
    "night":         "NIGHT 21–05",
}


# ─── DB helper ───────────────────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


# ─── Core matching logic ─────────────────────────────────────────────────────
def _is_hit(pred: dict, arrests: list) -> bool:
    """
    Return True if any arrest in the list satisfies all match criteria
    for the given prediction.
    """
    pred_date = datetime.strptime(pred["target_date"], "%Y-%m-%d").date()
    pred_dist = (pred["district"] or "").lower().strip()
    pred_hour = pred.get("predicted_hour")        # float or None
    pred_tw   = pred.get("predicted_time_window") or ""

    for arr in arrests:
        # ── 1. District match ──────────────────────────────────────────────
        if (arr.get("district") or "").lower().strip() != pred_dist:
            continue

        # ── 2. Date within ±2 days (48 h) ─────────────────────────────────
        try:
            arr_date = datetime.strptime(arr["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if abs((arr_date - pred_date).days) > 2:
            continue

        # ── 3. Time check ──────────────────────────────────────────────────
        arr_hour = arr.get("hour_decimal")
        arr_tw   = arr.get("time_window") or ""

        if pred_hour is not None and arr_hour is not None:
            # Both have numeric hour — require within 4 h (with midnight wrap)
            diff = abs(float(arr_hour) - float(pred_hour))
            diff = min(diff, 24.0 - diff)
            if diff > 4.0:
                continue          # wrong time slot — try next arrest
        elif pred_tw and arr_tw:
            # Fall back to time-window comparison
            if arr_tw != pred_tw:
                continue          # window mismatch
        # else: no usable time data on either side — district+date is enough

        return True   # all checks passed

    return False


# ─── Evaluate a date range ────────────────────────────────────────────────────
def evaluate_range(date_from: date, date_to: date, force: bool = False) -> dict:
    """
    Score predictions whose target_date falls in [date_from, date_to].
    Skips already-evaluated rows unless force=True.
    Returns a stats dict with both per-run and cumulative figures.
    """
    conn = _conn()

    extra = "" if force else " AND was_correct IS NULL"
    preds = [dict(r) for r in conn.execute(
        f"SELECT * FROM predictions "
        f"WHERE target_date >= ? AND target_date <= ?{extra}",
        (str(date_from), str(date_to))
    ).fetchall()]

    # Fetch arrests in a slightly wider window to catch edge cases
    ext_from = (date_from - timedelta(days=2)).strftime("%Y-%m-%d")
    ext_to   = (date_to   + timedelta(days=2)).strftime("%Y-%m-%d")
    arrests  = [dict(r) for r in conn.execute(
        "SELECT date, district, hour_decimal, time_window "
        "FROM arrests WHERE date >= ? AND date <= ?",
        (ext_from, ext_to)
    ).fetchall()]

    new_hits = new_misses = 0
    for pred in preds:
        correct    = 1 if _is_hit(pred, arrests) else 0
        new_hits   += correct
        new_misses += 1 - correct
        conn.execute(
            "UPDATE predictions SET was_correct = ? WHERE id = ?",
            (correct, pred["id"])
        )
    conn.commit()

    # Cumulative stats for the full range (includes previously scored rows)
    rows = conn.execute(
        "SELECT was_correct FROM predictions "
        "WHERE target_date >= ? AND target_date <= ? AND was_correct IS NOT NULL",
        (str(date_from), str(date_to))
    ).fetchall()
    conn.close()

    total_hits   = sum(1 for r in rows if r["was_correct"] == 1)
    total_misses = len(rows) - total_hits
    total        = total_hits + total_misses

    return {
        "date_from":          str(date_from),
        "date_to":            str(date_to),
        "evaluated_this_run": new_hits + new_misses,
        "new_hits":           new_hits,
        "new_misses":         new_misses,
        "total":              total,
        "hits":               total_hits,
        "misses":             total_misses,
        "hit_rate":           total_hits / total if total else 0.0,
    }


# ─── Weekly summary ───────────────────────────────────────────────────────────
def weekly_summary() -> dict:
    """
    Returns accuracy stats for this week (last 7 days) vs last week (8–14 days ago),
    plus district and time-window breakdowns.
    """
    today     = date.today()
    this_from = today - timedelta(days=7)
    last_from = today - timedelta(days=14)
    last_to   = today - timedelta(days=8)

    conn = _conn()

    def _rows(d_from, d_to):
        return [dict(r) for r in conn.execute(
            "SELECT district, predicted_time_window, was_correct "
            "FROM predictions "
            "WHERE target_date >= ? AND target_date <= ? AND was_correct IS NOT NULL",
            (str(d_from), str(d_to))
        ).fetchall()]

    this_rows = _rows(this_from, today)
    last_rows = _rows(last_from, last_to)
    conn.close()

    def _rate(rows):
        if not rows:
            return 0.0, 0, 0
        hits = sum(1 for r in rows if r["was_correct"] == 1)
        return hits / len(rows), hits, len(rows) - hits

    this_rate, this_h, this_m = _rate(this_rows)
    last_rate, last_h, last_m = _rate(last_rows)

    # District breakdown (this week only)
    dist: dict = {}
    for r in this_rows:
        d = r["district"] or "Unknown"
        if d not in dist:
            dist[d] = [0, 0]   # [hits, total]
        dist[d][1] += 1
        dist[d][0] += r["was_correct"]
    dist_stats = {
        d: {"hits": v[0], "total": v[1], "rate": v[0] / v[1]}
        for d, v in dist.items()
    }

    # Time-window breakdown (this week only)
    tw: dict = {}
    for r in this_rows:
        t = r["predicted_time_window"] or "unknown"
        if t not in tw:
            tw[t] = [0, 0]
        tw[t][1] += 1
        tw[t][0] += r["was_correct"]
    tw_stats = {
        t: {"hits": v[0], "total": v[1], "rate": v[0] / v[1]}
        for t, v in tw.items()
    }

    best_d  = max(dist_stats, key=lambda k: dist_stats[k]["rate"], default=None)
    worst_d = (
        min(dist_stats, key=lambda k: dist_stats[k]["rate"], default=None)
        if len(dist_stats) > 1 else None
    )
    best_t  = max(tw_stats, key=lambda k: tw_stats[k]["rate"], default=None)
    worst_t = (
        min(tw_stats, key=lambda k: tw_stats[k]["rate"], default=None)
        if len(tw_stats) > 1 else None
    )

    return {
        "this_week": {
            "from": str(this_from), "to": str(today),
            "hits": this_h, "misses": this_m, "rate": this_rate,
        },
        "last_week": {
            "from": str(last_from), "to": str(last_to),
            "hits": last_h, "misses": last_m, "rate": last_rate,
        },
        "delta":          this_rate - last_rate,
        "best_district":  (best_d,  dist_stats[best_d]["rate"])  if best_d  else None,
        "worst_district": (worst_d, dist_stats[worst_d]["rate"]) if worst_d else None,
        "best_tw":        (best_t,  tw_stats[best_t]["rate"])    if best_t  else None,
        "worst_tw":       (worst_t, tw_stats[worst_t]["rate"])   if worst_t else None,
        "district_stats": dist_stats,
        "tw_stats":       tw_stats,
    }


# ─── Manual override ──────────────────────────────────────────────────────────
def mark_prediction(pred_id: int, correct: bool):
    """Manually set was_correct for a specific prediction ID."""
    conn = _conn()
    row = conn.execute(
        "SELECT id, target_date, district FROM predictions WHERE id = ?",
        (pred_id,)
    ).fetchone()
    if not row:
        print(f"[ERROR] Prediction ID {pred_id} not found in database.")
        conn.close()
        return
    conn.execute(
        "UPDATE predictions SET was_correct = ? WHERE id = ?",
        (1 if correct else 0, pred_id)
    )
    conn.commit()
    conn.close()
    label = "HIT  ✓" if correct else "MISS ✗"
    print(f"[OK] Prediction #{pred_id}  {dict(row)['target_date']}  "
          f"{dict(row)['district']}  →  {label}")


# ─── Pretty printers ──────────────────────────────────────────────────────────
def _bar(rate: float, width: int = 28) -> str:
    filled = int(rate * width)
    return "█" * filled + "░" * (width - filled)


def print_scorecard(result: dict):
    rate = result["hit_rate"]
    sep  = "═" * 60
    print(f"\n{sep}")
    print(f"  ACCURACY SCORECARD   {result['date_from']} → {result['date_to']}")
    print(sep)
    print(f"  Evaluated this run   : {result['evaluated_this_run']}")
    print(f"  New hits             : {result['new_hits']}  ✓")
    print(f"  New misses           : {result['new_misses']}  ✗")
    print(f"{'─' * 60}")
    print(f"  TOTAL EVALUATED      : {result['total']}")
    print(f"  HITS                 : {result['hits']}")
    print(f"  MISSES               : {result['misses']}")
    print(f"  HIT RATE             : {rate:.1%}   {_bar(rate)}")
    print(f"{sep}\n")


def print_weekly_summary():
    s     = weekly_summary()
    tw    = s["this_week"]
    lw    = s["last_week"]
    delta = s["delta"]
    arrow = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "─")
    sep   = "═" * 60

    print(f"{sep}")
    print(f"  WEEKLY ACCURACY COMPARISON")
    print(f"{sep}")
    print(f"  THIS WEEK  {tw['from']} → {tw['to']}")
    print(f"    {tw['rate']:.1%}   ({tw['hits']} hits, {tw['misses']} misses)"
          f"   {_bar(tw['rate'], 20)}")
    print(f"  LAST WEEK  {lw['from']} → {lw['to']}")
    print(f"    {lw['rate']:.1%}   ({lw['hits']} hits, {lw['misses']} misses)"
          f"   {_bar(lw['rate'], 20)}")
    print(f"  CHANGE     {arrow}  {delta:+.1%}")
    print(f"{'─' * 60}")

    if s["best_district"]:
        d, r = s["best_district"]
        print(f"  BEST DISTRICT   : {d}  ({r:.0%})")
    if s["worst_district"]:
        d, r = s["worst_district"]
        print(f"  WORST DISTRICT  : {d}  ({r:.0%})")
    if s["best_tw"]:
        t, r = s["best_tw"]
        label = TW_LABELS.get(t, t)
        print(f"  BEST WINDOW     : {label}  ({r:.0%})")
    if s["worst_tw"]:
        t, r = s["worst_tw"]
        label = TW_LABELS.get(t, t)
        print(f"  WORST WINDOW    : {label}  ({r:.0%})")

    print(f"{sep}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        prog="accuracy.py",
        description="KKTC Prediction Accuracy Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd")

    # ── check subcommand ──────────────────────────────────────────────────
    chk = sub.add_parser("check", help="Evaluate predictions for a date range")
    chk.add_argument(
        "--from", dest="date_from", required=True, metavar="YYYY-MM-DD",
        help="Start of date range (inclusive)",
    )
    chk.add_argument(
        "--to", dest="date_to", required=True, metavar="YYYY-MM-DD",
        help="End of date range (inclusive)",
    )
    chk.add_argument(
        "--force", action="store_true",
        help="Re-evaluate predictions that were already scored",
    )

    # ── mark subcommand ───────────────────────────────────────────────────
    mrk = sub.add_parser("mark", help="Manually override a prediction result")
    mrk.add_argument("--prediction-id", type=int, required=True,
                     help="ID from the predictions table")
    mrk.add_argument("--correct", choices=["yes", "no"], required=True,
                     help="yes = HIT, no = MISS")

    args = ap.parse_args()

    if args.cmd == "check":
        try:
            d_from = datetime.strptime(args.date_from, "%Y-%m-%d").date()
            d_to   = datetime.strptime(args.date_to,   "%Y-%m-%d").date()
        except ValueError as e:
            print(f"[ERROR] Bad date format: {e}")
            sys.exit(1)
        if d_from > d_to:
            print("[ERROR] --from must be before or equal to --to")
            sys.exit(1)
        result = evaluate_range(d_from, d_to, force=args.force)
        print_scorecard(result)
        print_weekly_summary()

    elif args.cmd == "mark":
        mark_prediction(args.prediction_id, args.correct == "yes")

    else:
        # Default: evaluate last 7 days + show weekly summary
        today  = date.today()
        d_from = today - timedelta(days=7)
        result = evaluate_range(d_from, today)
        print_scorecard(result)
        print_weekly_summary()


if __name__ == "__main__":
    main()
