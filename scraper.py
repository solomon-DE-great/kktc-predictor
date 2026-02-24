"""
KKTC Arrest Prediction System — Web Scraper
Scrapes arrest-related articles from KKTC news sites.
Handles both backfill (historical) and incremental (new) scraping.
"""
import re
import sys
import time
import json
import hashlib
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, quote_plus

# Ensure Turkish characters print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import requests
from bs4 import BeautifulSoup

from config import (
    SOURCES, ARREST_KEYWORDS, ARREST_KEYWORDS_ALWAYS, ARREST_KEYWORDS_CONTEXT,
    ARREST_CONTEXT_TERMS, BONUS_KEYWORDS, TITLE_FETCH_KEYWORDS,
    MAX_PAGES_PER_SOURCE, REQUEST_DELAY_SECONDS, RAW_ARTICLES_DIR, SSL_UNVERIFIED_HOSTS
)
from database import get_connection, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/scraper.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}


def is_relevant_article(title: str, body: str = "") -> bool:
    """Two-tier relevance check — sets the relevant=1 flag (triggers Claude extraction).
    Always-keywords fire unconditionally; context-keywords require immigration context."""
    text = f"{title} {body}".lower()
    if any(kw in text for kw in ARREST_KEYWORDS_ALWAYS):
        return True
    if any(kw in text for kw in ARREST_KEYWORDS_CONTEXT):
        return any(ctx in text for ctx in ARREST_CONTEXT_TERMS)
    return False


def is_worth_fetching(title: str) -> bool:
    """Broad title pre-filter — decides whether to bother fetching the full article page.
    Fetches if title contains any always-keyword, or any context-keyword (body checked later)."""
    text = title.lower()
    if any(kw in text for kw in ARREST_KEYWORDS_ALWAYS):
        return True
    if any(kw in text for kw in ARREST_KEYWORDS_CONTEXT):
        return True
    return any(kw in text for kw in TITLE_FETCH_KEYWORDS)


def fetch_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    """Fetch and parse a web page with retry logic."""
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    verify = not any(h in host for h in SSL_UNVERIFIED_HOSTS)
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15, verify=verify)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt+1}/{retries} failed for {url}: {e}")
            time.sleep(REQUEST_DELAY_SECONDS * (attempt + 1))
    return None


def save_article(url: str, source: str, title: str, body: str, publish_date: str = None) -> bool:
    """Save article to database and raw file. Returns True if newly inserted."""
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT OR IGNORE INTO articles (url, source, title, body, publish_date, relevant)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (url, source, title, body, publish_date, 1 if is_relevant_article(title, body) else 0))
        conn.commit()

        if cursor.rowcount == 0:
            return False  # Already existed — genuine duplicate

        # Save raw JSON for debugging
        slug = hashlib.md5(url.encode()).hexdigest()[:12]
        raw_path = RAW_ARTICLES_DIR / f"{source}_{slug}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump({
                "url": url, "source": source, "title": title,
                "body": body, "publish_date": publish_date,
                "scraped_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        return True
    finally:
        conn.close()


def extract_article_body(soup: BeautifulSoup) -> str:
    """Try multiple selectors to extract main article body text."""
    for selector in [
        "article", ".news-detail", ".haber-detay", ".haber-icerik",
        ".news-content", ".entry-content", ".post-content", ".content",
        ".the-content", "#icerik",
    ]:
        el = soup.select_one(selector)
        if el:
            return el.get_text(separator="\n", strip=True)
    paragraphs = soup.find_all("p")
    return "\n".join(p.get_text(strip=True) for p in paragraphs)


def extract_article_date(soup: BeautifulSoup, body: str = "") -> str | None:
    """Try to extract publish date from page."""
    time_el = soup.find("time")
    if time_el:
        return time_el.get("datetime", time_el.get_text(strip=True))
    # Look for date patterns in body text
    date_match = re.search(r"(\d{1,2})[\./](\d{1,2})[\./](\d{4})", body)
    if date_match:
        d, m, y = date_match.groups()
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    return None


# ═══════════════════════════════════════════════════════════
# SOURCE-SPECIFIC SCRAPERS
# ═══════════════════════════════════════════════════════════

class KibrisPostasiScraper:
    """
    Primary source: kibrispostasi.com
    Archive: /c57-Adli_Haberler (primary), plus several local sections.
    Search: /ara?q=QUERY (searches entire site archive).
    """
    SOURCE = "kibrispostasi"
    BASE = "https://www.kibrispostasi.com"

    SECTIONS = [
        "/c57-Adli_Haberler",     # Crime news (primary)
        "/c35-KIBRIS_HABERLERI",  # General KKTC news
        "/c87-LEFKOSA",
        "/c86-GIRNE",
        "/c88-GAZIMAGUSA",
        "/c96-GONYELI",
        "/c98-ALAYKOY",
        "/c69-GUZELYURT",
    ]

    SEARCH_QUERIES = [
        "ikamet izinsiz",
        "tutuklandı kaçak",
        "muhaceret denetim",
        "gözaltı yasadışı",
        "sınır dışı edildi",
    ]

    def scrape_section(self, section_path: str, max_pages: int = 10) -> list[dict]:
        """Scrape one archive section, applying broad title filter before fetching."""
        articles = []
        for page in range(1, max_pages + 1):
            url = f"{self.BASE}{section_path}" + (f"?page={page}" if page > 1 else "")
            log.info(f"[{self.SOURCE}] Scraping page {page}: {url}")

            soup = fetch_page(url)
            if not soup:
                break

            links = soup.find_all("a", href=re.compile(r"/[cn]\d+-.*/n\d+-"))
            if not links:
                links = soup.find_all("a", href=re.compile(r"/n\d+-"))

            if not links:
                log.info(f"[{self.SOURCE}] No articles found on page {page}")
                break

            new_count = 0
            for link in links:
                href = link.get("href", "")
                title = link.get_text(strip=True)

                # Broad title pre-filter — catches "tutuklandı", "gözaltı", etc.
                if not is_worth_fetching(title):
                    continue

                full_url = urljoin(self.BASE, href)
                article = self._fetch_article(full_url)
                if article:
                    saved = save_article(
                        url=full_url,
                        source=self.SOURCE,
                        title=article["title"],
                        body=article["body"],
                        publish_date=article.get("date"),
                    )
                    if saved:
                        new_count += 1
                        articles.append(article)

                time.sleep(REQUEST_DELAY_SECONDS)

            log.info(f"[{self.SOURCE}] Page {page}: {new_count} new articles saved")
            if new_count == 0 and page > 3:
                break

        return articles

    def scrape_search(self, query: str = "ikamet izinsiz", max_pages: int = 10) -> list[dict]:
        """Search entire site archive via /ara?q= endpoint (no title filter needed)."""
        articles = []
        encoded = quote_plus(query)

        for page in range(1, max_pages + 1):
            if page == 1:
                url = f"{self.BASE}/ara?q={encoded}"
            else:
                url = f"{self.BASE}/ara?q={encoded}&page={page}"

            log.info(f"[{self.SOURCE}] Search '{query}' page {page}: {url}")
            soup = fetch_page(url)
            if not soup:
                break

            links = soup.find_all("a", href=re.compile(r"/n\d+-"))
            if not links:
                links = soup.find_all("a", href=re.compile(r"/[cn]\d+-.*/n\d+-"))

            if not links:
                log.info(f"[{self.SOURCE}] No search results on page {page}")
                break

            new_count = 0
            seen = set()
            for link in links:
                href = link.get("href", "")
                if href in seen:
                    continue
                seen.add(href)
                title = link.get_text(strip=True)
                if len(title) < 10:
                    continue

                full_url = urljoin(self.BASE, href)
                article = self._fetch_article(full_url)
                if article:
                    saved = save_article(
                        url=full_url,
                        source=self.SOURCE,
                        title=article["title"],
                        body=article["body"],
                        publish_date=article.get("date"),
                    )
                    if saved:
                        new_count += 1
                        articles.append(article)

                time.sleep(REQUEST_DELAY_SECONDS)

            log.info(f"[{self.SOURCE}] Search page {page}: {new_count} new articles")
            if new_count == 0 and page > 2:
                break

        return articles

    def _fetch_article(self, url: str) -> dict | None:
        soup = fetch_page(url)
        if not soup:
            return None
        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else ""
        body = extract_article_body(soup)
        date = extract_article_date(soup, body)
        return {"url": url, "title": title, "body": body, "date": date}

    def backfill(self, max_pages: int = None):
        """Full historical scrape: archive sections + site-wide search queries."""
        max_pages = max_pages or MAX_PAGES_PER_SOURCE
        all_articles = []

        for section in self.SECTIONS:
            log.info(f"[{self.SOURCE}] === Section: {section} ===")
            all_articles.extend(self.scrape_section(section, max_pages=max_pages))

        for query in self.SEARCH_QUERIES:
            log.info(f"[{self.SOURCE}] === Search: '{query}' ===")
            all_articles.extend(self.scrape_search(query, max_pages=max_pages))

        return all_articles

    def scrape_new(self, max_pages: int = 3):
        return self.scrape_section("/c57-Adli_Haberler", max_pages=max_pages)


class HaberKibrisScraper:
    """
    haberkibris.com — WordPress-based site.
    Article URL format: /SLUG-ARTICLEID-YYYY-MM-DD.html
    Search: /?s=QUERY&paged=N
    """
    SOURCE = "haberkibris"
    BASE = "https://haberkibris.com"

    SEARCH_QUERIES = [
        "ikamet izinsiz",
        "kaçak gözaltı",
        "muhaceret denetim",
        "tutuklandı yasadışı",
        "sınır dışı",
    ]

    def scrape_search(self, query: str = "ikamet izinsiz", max_pages: int = 10) -> list[dict]:
        articles = []
        encoded = quote_plus(query)

        for page in range(1, max_pages + 1):
            if page == 1:
                url = f"{self.BASE}/?s={encoded}"
            else:
                url = f"{self.BASE}/?s={encoded}&paged={page}"

            log.info(f"[{self.SOURCE}] Search '{query}' page {page}: {url}")
            soup = fetch_page(url)
            if not soup:
                break

            # haberkibris article links end in .html and live on haberkibris.com
            links = [
                a for a in soup.find_all("a", href=re.compile(r"\.html"))
                if "haberkibris.com" in a.get("href", "")
            ]
            # Fallback: WordPress entry-title links
            if not links:
                links = soup.select(".entry-title a, h2 a, h3 a")

            if not links and page > 1:
                break

            new_count = 0
            seen = set()
            for link in links:
                href = link.get("href", "")
                if href in seen:
                    continue
                seen.add(href)

                title = link.get_text(strip=True)
                if len(title) < 10:
                    continue

                full_url = href if href.startswith("http") else urljoin(self.BASE, href)
                soup2 = fetch_page(full_url)
                if not soup2:
                    continue

                h1 = soup2.find("h1")
                full_title = h1.get_text(strip=True) if h1 else title
                body = extract_article_body(soup2)
                date = extract_article_date(soup2, body)

                saved = save_article(full_url, self.SOURCE, full_title, body, date)
                if saved:
                    new_count += 1
                    articles.append({"url": full_url, "title": full_title, "body": body})

                time.sleep(REQUEST_DELAY_SECONDS)

            log.info(f"[{self.SOURCE}] Page {page}: {new_count} new articles")
            if new_count == 0 and page > 2:
                break

        return articles

    def backfill(self, max_pages: int = None):
        max_pages = max_pages or MAX_PAGES_PER_SOURCE
        articles = []
        for query in self.SEARCH_QUERIES:
            articles.extend(self.scrape_search(query, max_pages=max_pages))
        return articles

    def scrape_new(self, max_pages: int = 2):
        return self.scrape_search("ikamet izinsiz", max_pages=max_pages)


class YorumKibrisScraper:
    """
    yorumkibris.com — search endpoint returns 410.
    Strategy: scrape main page + paginate (/sayfa/N/).
    Article URL pattern: /haber/SLUG.html
    """
    SOURCE = "yorumkibris"
    BASE = "https://yorumkibris.com"

    def scrape_page(self, url: str) -> list[dict]:
        articles = []
        soup = fetch_page(url)
        if not soup:
            return articles

        links = soup.find_all("a", href=re.compile(r"/haber/.*\.html"))
        seen = set()
        for link in links:
            href = link.get("href", "")
            if href in seen:
                continue
            seen.add(href)

            title = link.get_text(strip=True)
            if len(title) < 10:
                continue
            if not is_worth_fetching(title):
                continue

            full_url = urljoin(self.BASE, href)
            soup2 = fetch_page(full_url)
            if not soup2:
                continue

            h1 = soup2.find("h1")
            full_title = h1.get_text(strip=True) if h1 else title
            body = extract_article_body(soup2)
            date = extract_article_date(soup2, body)

            saved = save_article(full_url, self.SOURCE, full_title, body, date)
            if saved:
                articles.append({"url": full_url, "title": full_title, "body": body})

            time.sleep(REQUEST_DELAY_SECONDS)

        return articles

    def backfill(self, max_pages: int = None):
        max_pages = max_pages or min(MAX_PAGES_PER_SOURCE, 50)
        all_articles = []

        for page in range(1, max_pages + 1):
            if page == 1:
                url = f"{self.BASE}/"
            else:
                url = f"{self.BASE}/sayfa/{page}/"

            log.info(f"[{self.SOURCE}] Page {page}: {url}")
            page_articles = self.scrape_page(url)
            all_articles.extend(page_articles)

            if not page_articles and page > 2:
                log.info(f"[{self.SOURCE}] No articles on page {page}, stopping")
                break

        return all_articles

    def scrape_new(self, max_pages: int = 2):
        return self.backfill(max_pages)


class WordPressSearchScraper:
    """
    Generic WordPress search scraper (/?s=QUERY&paged=N).
    Used for: inkilapci.com, kibrisgercek.com, kibristurk.com, kibrismanset.com
    """

    def __init__(self, source_name: str, base_url: str, queries: list[str] = None):
        self.source = source_name
        self.base = base_url.rstrip("/")
        self.queries = queries or ["ikamet izinsiz", "gözaltı kaçak", "muhaceret yasadışı"]

    def scrape_query(self, query: str, max_pages: int = 10) -> list[dict]:
        articles = []
        encoded = quote_plus(query)

        for page in range(1, max_pages + 1):
            if page == 1:
                url = f"{self.base}/?s={encoded}"
            else:
                url = f"{self.base}/?s={encoded}&paged={page}"

            log.info(f"[{self.source}] Search '{query}' page {page}: {url}")
            soup = fetch_page(url)
            if not soup:
                break

            # Try WordPress-standard article title selectors first
            article_links = []
            for sel in [
                ".entry-title a", "h2.entry-title a", "h3.entry-title a",
                ".post-title a", "article h2 a", "article h3 a",
                ".haber-baslik a", ".baslik a",
            ]:
                found = soup.select(sel)
                if found:
                    article_links = [(a.get("href", ""), a.get_text(strip=True)) for a in found]
                    break

            # Fallback: any link with substantial title pointing to same domain
            if not article_links:
                domain = self.base.split("//", 1)[1]
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "")
                    title = a.get_text(strip=True)
                    if (len(title) >= 20
                            and "?" not in href
                            and "#" not in href
                            and (href.startswith(self.base) or
                                 href.startswith("/") or
                                 domain in href)):
                        article_links.append((href, title))

            if not article_links:
                log.info(f"[{self.source}] No results on page {page}")
                break

            new_count = 0
            seen = set()
            for href, title in article_links:
                if href in seen:
                    continue
                seen.add(href)
                if len(title) < 10:
                    continue

                full_url = href if href.startswith("http") else urljoin(self.base, href)
                soup2 = fetch_page(full_url)
                if not soup2:
                    continue

                h1 = soup2.find("h1")
                full_title = h1.get_text(strip=True) if h1 else title
                body = extract_article_body(soup2)
                date = extract_article_date(soup2, body)

                saved = save_article(full_url, self.source, full_title, body, date)
                if saved:
                    new_count += 1
                    articles.append({"url": full_url, "title": full_title, "body": body})

                time.sleep(REQUEST_DELAY_SECONDS)

            log.info(f"[{self.source}] Page {page}: {new_count} new articles")
            if new_count == 0 and page > 2:
                break

        return articles

    def backfill(self, max_pages: int = None):
        max_pages = max_pages or MAX_PAGES_PER_SOURCE
        articles = []
        for query in self.queries:
            articles.extend(self.scrape_query(query, max_pages))
        return articles

    def scrape_new(self, max_pages: int = 2):
        return self.scrape_query(self.queries[0], max_pages)


class GenericNewsScraper:
    """Legacy generic scraper for sources not covered by the typed classes above."""

    def __init__(self, source_name: str, base_url: str, search_url: str):
        self.source = source_name
        self.base = base_url
        self.search_url = search_url

    def scrape_search(self, max_pages: int = 5) -> list[dict]:
        articles = []
        soup = fetch_page(self.search_url)
        if not soup:
            return articles

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            title = link.get_text(strip=True)
            if len(title) < 15 or not is_worth_fetching(title):
                continue

            full_url = urljoin(self.base, href)
            article_soup = fetch_page(full_url)
            if not article_soup:
                continue

            body = extract_article_body(article_soup)
            saved = save_article(full_url, self.source, title, body)
            if saved:
                articles.append({"url": full_url, "title": title, "body": body})

            time.sleep(REQUEST_DELAY_SECONDS)

        return articles

    def backfill(self, max_pages=None):
        return self.scrape_search(max_pages or 10)

    def scrape_new(self, max_pages=2):
        return self.scrape_search(max_pages)


# ═══════════════════════════════════════════════════════════
# MAIN SCRAPING FUNCTIONS
# ═══════════════════════════════════════════════════════════

def get_all_scrapers():
    """Initialize all scraper instances."""
    return [
        KibrisPostasiScraper(),
        HaberKibrisScraper(),
        YorumKibrisScraper(),
        WordPressSearchScraper(
            "inkilapci", "https://www.inkilapci.com",
            queries=["ikamet izinsiz", "gözaltı kaçak", "muhaceret yasadışı"],
        ),
        WordPressSearchScraper(
            "kibrisgercek", "https://www.kibrisgercek.com",
            queries=["ikamet izinsiz", "gözaltı kaçak", "muhaceret yasadışı"],
        ),
        WordPressSearchScraper(
            "kibristurk", "https://www.kibristurk.com",
            queries=["ikamet izinsiz", "gözaltı kaçak", "muhaceret yasadışı"],
        ),
        WordPressSearchScraper(
            "kibrismanset", "https://www.kibrismanset.com",
            queries=["ikamet izinsiz", "gözaltı kaçak", "muhaceret yasadışı"],
        ),
    ]


def backfill_all(max_pages: int = None):
    """Full historical scrape across all sources."""
    init_db()
    total = 0
    for scraper in get_all_scrapers():
        source_name = getattr(scraper, "SOURCE", None) or getattr(scraper, "source", "unknown")
        log.info(f"=== BACKFILL: {source_name} ===")
        try:
            articles = scraper.backfill(max_pages)
            total += len(articles)
            log.info(f"[{source_name}] Found {len(articles)} new relevant articles")
        except Exception as e:
            log.error(f"Error in backfill for {source_name}: {e}", exc_info=True)

    log.info(f"=== BACKFILL COMPLETE: {total} total new articles ===")
    return total


def scrape_new():
    """Incremental scrape — only recent articles."""
    init_db()
    total = 0
    for scraper in get_all_scrapers():
        source_name = getattr(scraper, "SOURCE", None) or getattr(scraper, "source", "unknown")
        log.info(f"--- Checking {source_name} for new articles ---")
        try:
            articles = scraper.scrape_new()
            total += len(articles)

            conn = get_connection()
            conn.execute("""
                INSERT INTO scrape_log (source, articles_found, new_articles)
                VALUES (?, ?, ?)
            """, (source_name, len(articles), len(articles)))
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"Error scraping {source_name}: {e}")

    log.info(f"--- Incremental scrape complete: {total} new articles ---")
    return total


def get_unprocessed_articles() -> list[dict]:
    """Get articles that haven't been processed by the extractor yet."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM articles WHERE processed = 0 AND relevant = 1
        ORDER BY scraped_at ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_article_processed(article_id: int, arrests_extracted: int = 0):
    """Mark an article as processed."""
    conn = get_connection()
    conn.execute("UPDATE articles SET processed = 1 WHERE id = ?", (article_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KKTC Arrest News Scraper")
    parser.add_argument("mode", choices=["backfill", "new"], help="Scraping mode")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages per source")
    args = parser.parse_args()

    if args.mode == "backfill":
        backfill_all(args.max_pages)
    else:
        scrape_new()