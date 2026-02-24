"""
KKTC Arrest Prediction System — Configuration
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "arrests.db"
RAW_ARTICLES_DIR = DATA_DIR / "raw_articles"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, RAW_ARTICLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Keys ────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_GEOCODING_KEY = os.getenv("GOOGLE_GEOCODING_KEY", "")

# ─── Scraping Config ─────────────────────────────────────
SCRAPE_INTERVAL_HOURS = 6
MAX_PAGES_PER_SOURCE = 200  # How many archive pages to crawl during backfill
REQUEST_DELAY_SECONDS = 2   # Be polite to servers

# Target news sources (all republish the same police bulletins)
SOURCES = {
    "kibrispostasi": {
        "base_url": "https://www.kibrispostasi.com",
        "archive_url": "https://www.kibrispostasi.com/c57-Adli_Haberler",
        "priority": 1,
    },
    "haberkibris": {
        "base_url": "https://haberkibris.com",
        "search_url": "https://haberkibris.com/?s=ikamet+izinsiz",
        "priority": 2,
    },
    "kibristurk": {
        "base_url": "https://www.kibristurk.com",
        "search_url": "https://www.kibristurk.com/?s=ikamet+izinsiz",
        "priority": 3,
    },
    "yorumkibris": {
        "base_url": "https://yorumkibris.com",
        # Search returns 410 — scraper uses main-page pagination instead
        "priority": 3,
    },
    "inkilapci": {
        "base_url": "https://www.inkilapci.com",
        "search_url": "https://www.inkilapci.com/?s=ikamet+izinsiz",
        "priority": 3,
    },
    "kibrisgercek": {
        "base_url": "https://www.kibrisgercek.com",
        "search_url": "https://www.kibrisgercek.com/?s=ikamet+izinsiz",
        "priority": 3,
    },
    "kibrismanset": {
        "base_url": "https://www.kibrismanset.com",
        "search_url": "https://www.kibrismanset.com/?s=ikamet+izinsiz",
        "priority": 3,
    },
}

# ─── Turkish Keywords for Filtering ──────────────────────
# Articles MUST contain at least one of these to be relevant
# Always relevant — no context required
ARREST_KEYWORDS_ALWAYS = [
    "ikamet izinsiz",
    "izinsiz ikamet",
    "muhaceret kontrol",
    "kaçak yaşam",
    "yasadışı ikamet",
    "ikamet izni olmayan",
    "izinsiz olarak ikamet",
    "izinsiz olarak kaldığı",
    "izinsiz olarak bulunduğu",
    "yasadışı giriş",
    "çalışma izinsiz",
    "kaçak işçi",
    "sınır dışı edilecek",
]

# Relevant only when the article ALSO mentions immigration/residence context
ARREST_KEYWORDS_CONTEXT = [
    "tutuklandı",
    "gözaltına alındı",
    "sınır dışı",
]
ARREST_CONTEXT_TERMS = ["ikamet", "muhaceret", "kaçak"]

# Combined list for any code that still needs a flat keyword list
ARREST_KEYWORDS = ARREST_KEYWORDS_ALWAYS + ARREST_KEYWORDS_CONTEXT

# Hosts with broken/self-signed SSL certs — scraper will skip verification for these only
SSL_UNVERIFIED_HOSTS = {
    "haberkibris.com",
    "yorumkibris.com",
    "kibrisgercek.com",
    "kibristurk.com",
    "kibrismanset.com",
    "kibrispostasi.com",
}

# Bonus keywords (deportation / work permit violations)
BONUS_KEYWORDS = [
    "ihraç edilecek",
]

# Broader title pre-filter keywords — used ONLY to decide whether to fetch an article.
# ARREST_KEYWORDS (strict) still determine the relevant=1 flag after body is fetched.
TITLE_FETCH_KEYWORDS = ARREST_KEYWORDS + [
    "tutuklandı",
    "gözaltına alındı",
    "sınır dışı",
    "muhaceret",
    "yasadışı",
    "kaçak",
]

# ─── Known Geocoding Cache ───────────────────────────────
# Pre-populated with locations from your existing data + common areas
# Format: "area, district" → (latitude, longitude)
KNOWN_LOCATIONS = {
    # Lefkoşa
    "İstanbul Caddesi, Lefkoşa": (35.1782, 33.3615),
    "Cemal Gürsel Caddesi, Lefkoşa": (35.1821, 33.3622),
    "Mehmet Akif Caddesi, Lefkoşa": (35.188, 33.351),
    "Atatürk Caddesi, Lefkoşa": (35.183, 33.355),
    "Surlariçi, Lefkoşa": (35.1753, 33.3616),
    "Arasta, Lefkoşa": (35.1748, 33.3620),
    "Lefkoşa Merkez": (35.1856, 33.3821),
    # Gönyeli
    "Atatürk Caddesi, Gönyeli": (35.2114, 33.3112),
    "Ulus Caddesi, Gönyeli": (35.2144, 33.3155),
    "Gönyeli Merkez": (35.2100, 33.3100),
    # Alayköy
    "Sanayi Bölgesi, Alayköy": (35.195, 33.275),
    "Alayköy Merkez": (35.195, 33.280),
    # Haspolat
    "Sanayi Bölgesi, Haspolat": (35.2133, 33.4321),
    "Haspolat Merkez": (35.2100, 33.4300),
    # Demirhan
    "Demirhan Merkez": (35.1815, 33.4682),
    # Güzelyurt
    "Bostancı, Güzelyurt": (35.1961, 32.9833),
    "Güzelyurt Merkez": (35.1975, 32.9870),
    # Girne
    "Girne Merkez": (35.3364, 33.3178),
    "Alsancak, Girne": (35.3300, 33.2800),
    "Çatalköy, Girne": (35.3200, 33.3900),
    "Lapta, Girne": (35.3400, 33.1700),
    # Gazimağusa
    "Gazimağusa Merkez": (35.1250, 33.9417),
    "Surlariçi, Gazimağusa": (35.1248, 33.9410),
    # İskele
    "İskele Merkez": (35.2868, 33.8800),
    # Lefke
    "Lefke Merkez": (35.1100, 32.8500),
}

# ─── Prediction Config ───────────────────────────────────
MIN_RECORDS_FOR_PREDICTION = 30
PREDICTION_HORIZON_DAYS = 7
CONFIDENCE_THRESHOLD = 0.4
RETRAIN_INTERVAL_HOURS = 24

# Time windows for classification
TIME_WINDOWS = {
    "early_morning": (5, 8),
    "morning": (8, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 5),
}

# Districts for spatial classification
DISTRICTS = [
    "Lefkoşa", "Gönyeli", "Alayköy", "Haspolat", "Demirhan",
    "Güzelyurt", "Girne", "Gazimağusa", "İskele", "Lefke",
    "Alsancak", "Çatalköy", "Lapta", "Dikmen", "Akdoğan",
    "Değirmenlik", "Geçitkale", "Yeniboğaziçi",
]
