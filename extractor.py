"""
KKTC Arrest Prediction System — Data Extractor
Uses Claude API to extract structured arrest data from Turkish news articles.
"""
import json
import logging
import time
from datetime import datetime

import anthropic

from config import ANTHROPIC_API_KEY, KNOWN_LOCATIONS
from database import insert_arrest, get_connection
from scraper import get_unprocessed_articles, mark_article_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a precise data extraction system for North Cyprus (KKTC) police arrest reports about illegal immigration/residence. 

Extract EVERY distinct arrest event from the article. A single article often contains MULTIPLE events at different times, locations, or dates — extract each one separately.

For each event, extract:
- date: In YYYY-MM-DD format. Convert Turkish dates (e.g., "18 Şubat 2026" → "2026-02-18")
- time: In HH:MM format (e.g., "saat 21:00 raddelerinde" → "21:00", "saat 10.30 sıralarında" → "10:30")
- district: The main district/area (e.g., Lefkoşa, Gönyeli, Girne, Gazimağusa, Güzelyurt, Alayköy, Haspolat, Demirhan, İskele, Lefke)
- specific_area: The specific street or zone (e.g., "İstanbul Caddesi", "Sanayi Bölgesi", "Atatürk Caddesi", "Surlariçi")
- count: Number of people arrested in this specific event (count individuals)
- reason: The legal reason (usually "ikamet izinsiz" / "çalışma izinsiz" / "yasadışı giriş")
- operating_unit: Police unit if mentioned (e.g., "Cürümleri Önleme Şubesi", "Gönyeli Polis Karakolu")

CRITICAL RULES:
1. If ONE article mentions arrests at DIFFERENT streets, split them into separate events
2. If arrests happened on DIFFERENT dates, split them into separate events
3. If arrests happened at DIFFERENT times, split them into separate events
4. Count individuals carefully — "4 kişi" = count 4, not the number of names listed
5. When exact count isn't clear, count the names/initials mentioned
6. Ignore court hearing details (mahkeme, yargıç, tutukluluk süresi) — only extract the ORIGINAL arrest event
7. When a total count is given AND individual events are described, use the individual events (they have better detail)

Return ONLY a JSON array. No markdown, no explanation. Empty array [] if no arrest events found.

Example output:
[
  {
    "date": "2026-02-18",
    "time": "21:00",
    "district": "Lefkoşa",
    "specific_area": "İstanbul Caddesi",
    "count": 4,
    "reason": "ikamet izinsiz",
    "operating_unit": "Cürümleri Önleme Şubesi"
  },
  {
    "date": "2026-02-18",
    "time": "10:30",
    "district": "Gönyeli",
    "specific_area": "Atatürk Caddesi",
    "count": 2,
    "reason": "ikamet izinsiz",
    "operating_unit": "Cürümleri Önleme Şubesi"
  }
]"""


def extract_arrests_from_text(article_text: str, article_url: str = "") -> list[dict]:
    """Use Claude API to extract structured arrest data from article text."""
    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY not set! Run: export ANTHROPIC_API_KEY=your-key")
        return []

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            system=EXTRACTION_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Extract arrest events from this article:\n\n{article_text[:6000]}"
            }]
        )

        raw_text = response.content[0].text.strip()

        # Clean potential markdown code fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

        events = json.loads(raw_text)
        log.info(f"Extracted {len(events)} events from article")
        return events

    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}\nRaw: {raw_text[:200]}")
        return []
    except anthropic.APIError as e:
        log.error(f"Anthropic API error: {e}")
        return []
    except Exception as e:
        log.error(f"Extraction error: {e}")
        return []


def geocode_event(event: dict) -> tuple[float | None, float | None]:
    """Look up coordinates for an event location."""
    area = event.get("specific_area", "")
    district = event.get("district", "")

    # Try exact match first
    key = f"{area}, {district}"
    if key in KNOWN_LOCATIONS:
        return KNOWN_LOCATIONS[key]

    # Try partial matches
    for known_key, coords in KNOWN_LOCATIONS.items():
        if area and area in known_key:
            return coords
        if district and district in known_key and "Merkez" in known_key:
            return coords

    # Fallback: district center
    for known_key, coords in KNOWN_LOCATIONS.items():
        if district in known_key:
            return coords

    log.warning(f"No coordinates found for: {area}, {district}")
    return None, None


def process_article(article: dict) -> int:
    """Process one article: extract events, geocode, store. Returns count of new arrests."""
    body = article.get("body", "")
    title = article.get("title", "")
    url = article.get("url", "")
    full_text = f"{title}\n\n{body}"

    if len(full_text.strip()) < 50:
        log.warning(f"Article too short, skipping: {url}")
        return 0

    events = extract_arrests_from_text(full_text, url)
    new_count = 0

    for event in events:
        # Geocode
        lat, lng = geocode_event(event)
        event["latitude"] = lat
        event["longitude"] = lng
        event["source_url"] = url
        event["source_name"] = article.get("source", "unknown")
        event["raw_text"] = full_text[:500]  # Store excerpt for reference

        if insert_arrest(event):
            new_count += 1
            log.info(
                f"  ✓ New: {event.get('date')} {event.get('time')} "
                f"{event.get('district')}/{event.get('specific_area')} "
                f"({event.get('count')} people)"
            )

    return new_count


def process_all_unprocessed(batch_size: int = 50, delay: float = 1.0):
    """Process all unprocessed articles in the database."""
    articles = get_unprocessed_articles()
    log.info(f"Found {len(articles)} unprocessed relevant articles")

    total_arrests = 0
    for i, article in enumerate(articles[:batch_size]):
        log.info(f"[{i+1}/{min(len(articles), batch_size)}] Processing: {article.get('title', '')[:60]}...")

        count = process_article(article)
        mark_article_processed(article["id"], count)
        total_arrests += count

        time.sleep(delay)  # Rate limit API calls

    log.info(f"=== Processing complete: {total_arrests} new arrests extracted ===")
    return total_arrests


def extract_from_raw_text(text: str, source_url: str = "manual") -> int:
    """Directly extract and store from raw text (useful for manual input)."""
    events = extract_arrests_from_text(text, source_url)
    new_count = 0
    for event in events:
        lat, lng = geocode_event(event)
        event["latitude"] = lat
        event["longitude"] = lng
        event["source_url"] = source_url
        event["source_name"] = "manual"
        if insert_arrest(event):
            new_count += 1
    return new_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract arrest data from articles")
    parser.add_argument("--batch", type=int, default=50, help="Max articles to process")
    parser.add_argument("--text", type=str, help="Extract from raw text directly")
    args = parser.parse_args()

    if args.text:
        count = extract_from_raw_text(args.text)
        print(f"Extracted {count} arrests from text")
    else:
        process_all_unprocessed(args.batch)
