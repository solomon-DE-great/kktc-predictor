import os, sys

# Fix Windows Unicode
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"
# Load API key from .env file (never hardcode keys in source)
from pathlib import Path as _Path
_env = _Path(__file__).parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        if _l.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = _l.split("=", 1)[1].strip().strip('"')
            break

# Fix logging for Turkish characters
import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False))]
)

# Fix the geocoder bug (None district crash)
import extractor
original_process = extractor.process_article
def safe_process(article):
    try:
        events = extractor.extract_arrests_from_text(
            f"{article.get('title','')}\n\n{article.get('body','')}",
            article.get('url','')
        )
        count = 0
        for event in events:
            if not event.get("district"):
                event["district"] = "Unknown"
            if not event.get("specific_area"):
                event["specific_area"] = ""
            lat, lng = None, None
            try:
                lat, lng = extractor.geocode_event(event)
            except:
                pass
            event["latitude"] = lat
            event["longitude"] = lng
            event["source_url"] = article.get("url", "")
            event["source_name"] = article.get("source", "unknown")
            from database import insert_arrest
            if insert_arrest(event):
                count += 1
                print(f"  NEW: {event.get('date')} {event.get('time')} {event.get('district')}/{event.get('specific_area')} ({event.get('count')} people)")
        return count
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0

extractor.process_article = safe_process

# Reset articles
from database import get_connection, get_arrest_count
conn = get_connection()
conn.execute("UPDATE articles SET processed = 0 WHERE relevant = 1")
conn.commit()
ready = conn.execute("SELECT COUNT(*) FROM articles WHERE processed = 0 AND relevant = 1").fetchone()[0]
print(f"Ready to process: {ready}")
print(f"Records BEFORE: {get_arrest_count()}")
conn.close()

# Run extraction
extractor.process_all_unprocessed()

# Show final count
print(f"\nRecords AFTER: {get_arrest_count()}")