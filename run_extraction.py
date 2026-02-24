"""
Run extraction on all unprocessed relevant articles, then predictions.
"""
import os, sys

# Unicode fix for Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load API key from environment or .env file (never hardcode keys)
from pathlib import Path as _Path
_env_file = _Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = _line.split("=", 1)[1].strip().strip('"')
            break

import logging
import time

# Reconfigure logging to avoid unicode issues on Windows
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)
    )]
)
log = logging.getLogger(__name__)

import extractor
from database import insert_arrest, get_connection, get_arrest_count
from scraper import get_unprocessed_articles, mark_article_processed

# ── Patch process_article to be safe with None district ──────────────────────
def safe_process(article):
    body = article.get("body", "")
    title = article.get("title", "")
    url = article.get("url", "")
    full_text = f"{title}\n\n{body}"
    if len(full_text.strip()) < 50:
        return 0
    try:
        events = extractor.extract_arrests_from_text(full_text, url)
    except Exception as e:
        print(f"  ERROR extracting: {e}")
        return 0
    count = 0
    for event in events:
        if not event.get("district"):
            event["district"] = "Unknown"
        if not event.get("specific_area"):
            event["specific_area"] = ""
        lat, lng = None, None
        try:
            lat, lng = extractor.geocode_event(event)
        except Exception:
            pass
        event["latitude"] = lat
        event["longitude"] = lng
        event["source_url"] = url
        event["source_name"] = article.get("source", "unknown")
        event["raw_text"] = full_text[:500]
        try:
            if insert_arrest(event):
                count += 1
                print(f"  NEW: {event.get('date')} {event.get('time','?')} "
                      f"{event.get('district')}/{event.get('specific_area')} "
                      f"({event.get('count','?')} people)")
        except Exception as e:
            print(f"  DB ERROR: {e}")
    return count

extractor.process_article = safe_process

# ── Run extraction ────────────────────────────────────────────────────────────
conn = get_connection()
unprocessed = conn.execute(
    "SELECT COUNT(*) FROM articles WHERE processed=0 AND relevant=1"
).fetchone()[0]
before = get_arrest_count()
conn.close()

print(f"\n{'='*60}")
print(f"Unprocessed relevant articles: {unprocessed}")
print(f"Arrests BEFORE extraction:     {before}")
print(f"{'='*60}\n")

articles = get_unprocessed_articles()
total_new = 0
for i, article in enumerate(articles):
    title_preview = (article.get("title") or "")[:70]
    print(f"[{i+1}/{len(articles)}] {title_preview}...")
    sys.stdout.flush()
    count = safe_process(article)
    mark_article_processed(article["id"], count)
    total_new += count
    time.sleep(0.8)   # ~0.8 s between API calls

after = get_arrest_count()
print(f"\n{'='*60}")
print(f"Extraction complete!")
print(f"New arrests extracted:  {total_new}")
print(f"Arrests AFTER:          {after}")
print(f"{'='*60}\n")

# ── Run predictions ───────────────────────────────────────────────────────────
print("Running predictions...")
sys.stdout.flush()
from predictor import MasterPredictor
mp = MasterPredictor()
results = mp.generate_full_prediction(horizon_days=7, top_n=20)

# ── Show top 10 predictions ───────────────────────────────────────────────────
predictions = results.get("predictions", [])
print(f"\n{'='*60}")
print(f"FINAL ARREST COUNT IN DB: {after}")
print(f"{'='*60}")
print(f"\nTOP 10 PREDICTIONS (next 7 days):\n")
for rank, p in enumerate(predictions[:10], 1):
    date = p.get("target_date", "?")
    district = p.get("district", "?")
    area = p.get("specific_area", "") or ""
    tw = p.get("predicted_time_window", "?")
    hour = p.get("predicted_hour")
    hour_str = f"{int(hour):02d}:{int((hour%1)*60):02d}" if hour is not None else "?"
    cnt = p.get("predicted_count", "?")
    conf = p.get("confidence", 0)
    loc = f"{district}" + (f" / {area}" if area else "")
    print(f"  #{rank:2d}  {date}  {loc}")
    print(f"        Time: {tw} (~{hour_str})  Count: {cnt}  Confidence: {conf:.0%}")
    print()

# Also show daily forecast summary
forecast = results.get("daily_forecast", [])
if forecast:
    print(f"\n7-DAY DAILY FORECAST:")
    for day in forecast[:7]:
        print(f"  {day.get('date','?')}  ~{day.get('predicted_count', day.get('yhat','?')):.1f} arrests")
