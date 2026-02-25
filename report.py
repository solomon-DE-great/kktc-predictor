"""Generate a detailed 7-day prediction report from saved predictions.json"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from datetime import datetime, timedelta, date
from collections import defaultdict

data           = json.loads(Path("data/predictions.json").read_text(encoding="utf-8"))
predictions    = data.get("predictions", [])
daily_forecast = data.get("daily_forecast", [])
repeat_pats    = data.get("repeat_patterns", [])
op_pats        = data.get("operation_patterns", {})
data_stats     = data.get("data_stats", {})

TW = {
    "early_morning": "05:00 - 08:00  (Early Morning)",
    "morning":       "08:00 - 12:00  (Morning)",
    "afternoon":     "12:00 - 17:00  (Afternoon)",
    "evening":       "17:00 - 21:00  (Evening)",
    "night":         "21:00 - 05:00  (Night)",
}
TW_SHORT = {
    "early_morning": "05:00-08:00",
    "morning":       "08:00-12:00",
    "afternoon":     "12:00-17:00",
    "evening":       "17:00-21:00",
    "night":         "21:00-05:00",
}

def conf_label(c):
    return "HIGH" if c >= 0.6 else "MODERATE" if c >= 0.4 else "LOW"

def conf_stars(c):
    return "***" if c >= 0.6 else "** " if c >= 0.4 else "*  "

today = date.today()

# ── separate in-window vs outside-window predictions ──────────────────────
by_date = defaultdict(list)
for p in predictions:
    raw = p.get("predicted_date") or p.get("target_date", "")
    try:
        pd_ = datetime.strptime(raw[:10], "%Y-%m-%d").date()
        if today <= pd_ <= today + timedelta(days=7):
            by_date[pd_].append(p)
    except Exception:
        pass

W = 72

def divider(ch="─"):
    print("  " + ch * (W - 2))

def section(title):
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)

# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("  KKTC ENFORCEMENT INTELLIGENCE SYSTEM")
print("  7-DAY OPERATIONAL PREDICTION REPORT")
print(f"  Report Date : {today.strftime('%A, %d %B %Y')}")
print(f"  Period      : {today.strftime('%d %b %Y')}  to  {(today + timedelta(days=7)).strftime('%d %b %Y')}")
print(f"  Database    : {data_stats.get('total_records', 84)} arrest records")
print(f"  Model note  : {data_stats.get('model_confidence_note', 'MODERATE')}")
print("=" * W)

# ── Section 1: Daily Volume Forecast ─────────────────────────────────────
section("SECTION 1 — DAILY ARREST VOLUME FORECAST")
print(f"  {'DATE':<16} {'DAY':<11} {'FORECAST':>9}  {'RANGE':<22}  CHART")
divider()
total_7d = 0
for day in daily_forecast[:7]:
    d = day.get("date", "?")
    try:
        dobj     = datetime.strptime(d, "%Y-%m-%d")
        day_name = dobj.strftime("%A")
        d_fmt    = dobj.strftime("%d %b %Y")
    except Exception:
        day_name, d_fmt = "?", d
    cnt = day.get("predicted_count", day.get("yhat", 0))
    hi  = day.get("upper_bound", "")
    lo  = day.get("lower_bound", "")
    rng = f"[{lo:.1f} - {hi:.1f}]" if hi != "" and lo != "" else "—"
    bar = "|" * min(int(cnt), 20)
    alert = "  !! ELEVATED" if cnt >= 10 else ""
    print(f"  {d_fmt:<16} {day_name:<11} {cnt:>7.1f}p  {rng:<22}  {bar}{alert}")
    total_7d += cnt
divider()
print(f"  {'7-DAY TOTAL ESTIMATE':<28} {total_7d:>7.1f} persons")

# ── Section 2: Per-Day Location Predictions ───────────────────────────────
section("SECTION 2 — LOCATION PREDICTIONS BY DAY")

forecast_dates = []
for d in daily_forecast[:7]:
    try:
        forecast_dates.append(datetime.strptime(d["date"], "%Y-%m-%d").date())
    except Exception:
        pass

for target_date in sorted(forecast_dates):
    day_name  = target_date.strftime("%A").upper()
    d_fmt     = target_date.strftime("%d %B %Y")
    day_preds = sorted(by_date.get(target_date, []),
                       key=lambda p: -p.get("confidence", 0))
    fc_cnt    = next(
        (d.get("predicted_count", d.get("yhat", 0))
         for d in daily_forecast
         if d.get("date", "") == target_date.strftime("%Y-%m-%d")),
        0,
    )
    alert = "  !! ELEVATED ACTIVITY" if fc_cnt >= 10 else ""
    print()
    print(f"  +-- {day_name}, {d_fmt}  (forecast: ~{fc_cnt:.1f} arrests){alert}")
    divider("-")

    if not day_preds:
        print("  |  No location-specific predictions for this date.")
        print("  |  Activity is expected but no dominant hotspot identified.")
        print()
        continue

    for i, p in enumerate(day_preds, 1):
        conf     = p.get("confidence", 0)
        district = (p.get("district") or "Unknown").upper()
        area     = (p.get("specific_area") or "").strip()
        tw_key   = p.get("predicted_time_window", "")
        tw_full  = TW.get(tw_key, tw_key or "Unknown")
        tw_s     = TW_SHORT.get(tw_key, "?")
        ph       = p.get("predicted_hour")
        hstr     = f"{int(ph):02d}:{int((ph % 1) * 60):02d}" if ph is not None else "?"
        cnt      = p.get("predicted_count", "?")
        lat      = p.get("latitude")
        lon      = p.get("longitude")
        model    = (p.get("model") or "unknown").upper()
        is_rep   = p.get("has_temporal_pattern", False)
        intv     = p.get("pattern_interval_days")
        hist_ev  = p.get("historical_events", 0)
        stars    = conf_stars(conf)
        clabel   = conf_label(conf)

        print(f"  |")
        print(f"  |  [{i}] {stars}  CONFIDENCE: {conf:.0%} ({clabel})")
        print(f"  |      District     : {district}")
        if area:
            print(f"  |      Specific Area: {area}")
        print(f"  |      Time Window  : {tw_full}")
        print(f"  |      Peak Hour    : ~{hstr}")
        print(f"  |      Est. Count   : ~{cnt} person(s)")
        if lat and lon:
            print(f"  |      Coordinates  : {lat:.4f}N  {lon:.4f}E")
        print(f"  |      Prev. Events : {hist_ev} prior arrest(s) at this location")
        print(f"  |      Model Used   : {model}")
        if is_rep and intv:
            print(f"  |      !! REPEAT CYCLE: operation every ~{intv:.1f} days")
    print()

# ── Section 3: Repeat Raid Alerts ─────────────────────────────────────────
section("SECTION 3 — REPEAT RAID CYCLE ALERTS")
if repeat_pats:
    for i, pat in enumerate(repeat_pats, 1):
        dist  = pat.get("district", "?").upper()
        area  = (pat.get("specific_area") or "").strip().upper()
        loc   = dist + (f" / {area}" if area else "")
        avg   = pat.get("avg_interval_days", "?")
        last  = pat.get("last_raid_date", "?")
        nxt   = pat.get("predicted_next_date", "?")
        cons  = pat.get("consistency_score", 0)
        n     = pat.get("event_count", "?")
        tw_k  = pat.get("time_window", "")
        tw    = TW.get(tw_k, tw_k or "Not specified")
        bar   = "|" * int(cons * 10) + "." * (10 - int(cons * 10))
        print()
        print(f"  [{i}] {loc}")
        print(f"       Total Raids     : {n} documented operations")
        print(f"       Avg Interval    : every ~{avg} days")
        print(f"       Last Operation  : {last}")
        print(f"       Next Estimated  : {nxt}")
        print(f"       Reliability     : {cons:.0%}  [{bar}]")
        if tw_k:
            print(f"       Time Window     : {tw}")
else:
    print("  No repeat patterns detected.")

# ── Section 4: Operation Day Preference ───────────────────────────────────
section("SECTION 4 — HISTORICAL OPERATION DAY PREFERENCE")
pref = op_pats.get("preferred_days", {})
if pref:
    max_ops = max(pref.values()) if pref else 1
    print()
    for day_name, ops in sorted(pref.items(), key=lambda x: -x[1]):
        bar = "|" * int((ops / max_ops) * 20)
        pct = ops / sum(pref.values()) * 100
        print(f"  {day_name:<12}  {bar:<20}  {ops} operation(s)  ({pct:.0f}%)")
else:
    print("  No operation day preference data available.")

# ── Section 5: Top 10 Priority Targets ────────────────────────────────────
section("SECTION 5 — TOP 10 PRIORITY TARGETS (ALL UPCOMING DATES)")
all_upcoming = []
for p in predictions:
    raw = p.get("predicted_date") or p.get("target_date", "")
    try:
        pd_ = datetime.strptime(raw[:10], "%Y-%m-%d").date()
        if today <= pd_ <= today + timedelta(days=7):
            all_upcoming.append(p)
    except Exception:
        pass
all_upcoming.sort(key=lambda p: -p.get("confidence", 0))
print()
print(f"  {'#':<4} {'DATE':<13} {'DAY':<5} {'LOCATION':<36} {'WINDOW':<13} {'EST':>5} {'CONF':>6}")
divider()
for rank, p in enumerate(all_upcoming[:10], 1):
    raw      = p.get("predicted_date") or p.get("target_date", "?")
    try:
        dobj  = datetime.strptime(raw[:10], "%Y-%m-%d")
        d_fmt = dobj.strftime("%d %b %Y")
        dow   = dobj.strftime("%a").upper()
    except Exception:
        d_fmt, dow = raw, "?"
    district = (p.get("district") or "?").upper()
    area     = (p.get("specific_area") or "").strip()
    loc      = (district + (f"/{area}" if area else ""))[:35]
    tw_key   = p.get("predicted_time_window", "")
    tw_s     = TW_SHORT.get(tw_key, "?")
    cnt      = p.get("predicted_count", "?")
    conf     = p.get("confidence", 0)
    rep      = " R" if p.get("has_temporal_pattern") else ""
    print(f"  {rank:<4} {d_fmt:<13} {dow:<5} {loc:<36} {tw_s:<13} {str(cnt)+'p':>5} {conf:.0%}{rep}")
divider()
print("  R = Repeat pattern detected")

# ── Executive Summary ──────────────────────────────────────────────────────
section("EXECUTIVE SUMMARY")
best_day  = max(daily_forecast[:7], key=lambda d: d.get("predicted_count", 0))
bd_fmt    = datetime.strptime(best_day["date"], "%Y-%m-%d").strftime("%A %d %B")
bd_cnt    = best_day.get("predicted_count", 0)
reps_high = [r for r in repeat_pats if r.get("consistency_score", 0) >= 0.5]
print()
print(f"  Reporting period  : {today.strftime('%d %b')} - {(today+timedelta(days=7)).strftime('%d %b %Y')}")
print(f"  7-day est. total  : ~{total_7d:.0f} arrests")
print(f"  Peak day          : {bd_fmt}  (~{bd_cnt:.0f} arrests)")
if all_upcoming:
    tp   = all_upcoming[0]
    td   = (tp.get("district") or "?").upper()
    ta   = (tp.get("specific_area") or "").strip()
    tloc = td + (f" / {ta}" if ta else "")
    print(f"  Top priority      : {tloc}")
    print(f"  Confidence        : {tp.get('confidence',0):.0%}  ({conf_label(tp.get('confidence',0))})")
    print(f"  Time window       : {TW.get(tp.get('predicted_time_window',''), '?')}")
if reps_high:
    print(f"  High-reliability repeat sites ({len(reps_high)}):")
    for r in reps_high:
        d  = (r.get("district") or "?").upper()
        a  = (r.get("specific_area") or "").strip()
        lo = d + (f" / {a}" if a else "")
        print(f"    - {lo}  (next est: {r.get('predicted_next_date','?')}  reliability: {r.get('consistency_score',0):.0%})")
print()
print("=" * W)
print("  END OF REPORT  |  KKTC-INTEL  |  FOR OPERATIONAL USE ONLY")
print("=" * W)
