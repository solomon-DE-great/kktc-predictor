"""
KKTC Arrest Prediction System — Main Runner
Complete pipeline: Scrape → Extract → Predict → Export
"""
import argparse
import logging
import schedule
import time
from datetime import datetime
from pathlib import Path

from config import SCRAPE_INTERVAL_HOURS, RETRAIN_INTERVAL_HOURS, DATA_DIR
from database import init_db, import_csv, get_arrest_count, export_to_json
from scraper import backfill_all, scrape_new
from extractor import process_all_unprocessed
from predictor import MasterPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def run_full_pipeline():
    """Complete pipeline: scrape new → extract → predict → export."""
    log.info("=" * 60)
    log.info(f"PIPELINE START: {datetime.now().isoformat()}")
    log.info("=" * 60)

    # Step 1: Scrape new articles
    log.info("[1/4] Scraping new articles...")
    new_articles = scrape_new()
    log.info(f"  → {new_articles} new articles found")

    # Step 2: Extract arrest data from articles
    log.info("[2/4] Extracting arrest data with Claude API...")
    new_arrests = process_all_unprocessed()
    log.info(f"  → {new_arrests} new arrests extracted")

    # Step 3: Generate predictions
    log.info("[3/4] Running prediction models...")
    predictor = MasterPredictor()
    result = predictor.generate_full_prediction()
    log.info(f"  → {len(result['predictions'])} predictions generated")

    # Step 4: Export for dashboard
    log.info("[4/4] Exporting dashboard data...")
    path = export_to_json()
    log.info(f"  → Data exported to {path}")

    total = get_arrest_count()
    log.info(f"PIPELINE COMPLETE: {total} total records in database")
    log.info("=" * 60)

    return result


def run_initial_setup(csv_path: str = None):
    """First-time setup: init DB, import existing data, backfill, extract."""
    log.info("=== INITIAL SETUP ===")

    # Initialize database
    init_db()

    # Import existing CSV if provided
    if csv_path and Path(csv_path).exists():
        log.info(f"Importing existing CSV: {csv_path}")
        count = import_csv(csv_path)
        log.info(f"  → Imported {count} records from CSV")

    total = get_arrest_count()
    log.info(f"Database has {total} records after import")

    # Run initial prediction on existing data
    if total > 0:
        log.info("Running initial predictions on existing data...")
        predictor = MasterPredictor()
        print(predictor.quick_summary())

    return total


def run_backfill():
    """Historical backfill: scrape archives, extract, predict."""
    log.info("=== HISTORICAL BACKFILL ===")
    init_db()

    log.info("[1/3] Backfilling historical articles...")
    total_articles = backfill_all(max_pages=50)
    log.info(f"  → {total_articles} articles scraped")

    log.info("[2/3] Extracting arrest data...")
    total_arrests = process_all_unprocessed(batch_size=500, delay=0.5)
    log.info(f"  → {total_arrests} arrests extracted")

    log.info("[3/3] Generating predictions...")
    predictor = MasterPredictor()
    result = predictor.generate_full_prediction()
    print(predictor.quick_summary())

    export_to_json()
    log.info("=== BACKFILL COMPLETE ===")


def run_auto(interval_hours: int = None):
    """Run continuously with scheduled scraping and prediction updates."""
    interval = interval_hours or SCRAPE_INTERVAL_HOURS
    log.info(f"Starting auto-mode: pipeline every {interval} hours")

    # Run immediately
    run_full_pipeline()

    # Schedule recurring runs
    schedule.every(interval).hours.do(run_full_pipeline)

    log.info(f"Next run scheduled in {interval} hours. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)


def predict_only():
    """Just run predictions on existing data."""
    init_db()
    predictor = MasterPredictor()
    print(predictor.quick_summary())
    result = predictor.generate_full_prediction()
    export_to_json()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KKTC Arrest Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time — import your CSV and see initial predictions:
  python run.py setup --csv data/arrests.csv

  # Backfill historical data from news archives:
  python run.py backfill

  # Run the full pipeline once (scrape → extract → predict):
  python run.py pipeline

  # Run predictions only (no scraping):
  python run.py predict

  # Auto-mode: run pipeline every 6 hours:
  python run.py auto --interval 6
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Setup
    sp = subparsers.add_parser("setup", help="Initial setup + CSV import")
    sp.add_argument("--csv", type=str, help="Path to existing CSV file")

    # Backfill
    subparsers.add_parser("backfill", help="Historical backfill from news archives")

    # Pipeline
    subparsers.add_parser("pipeline", help="Run full pipeline once")

    # Predict
    subparsers.add_parser("predict", help="Run predictions on existing data")

    # Auto
    sp = subparsers.add_parser("auto", help="Continuous auto-mode")
    sp.add_argument("--interval", type=int, default=6, help="Hours between runs")

    args = parser.parse_args()

    if args.command == "setup":
        run_initial_setup(args.csv)
    elif args.command == "backfill":
        run_backfill()
    elif args.command == "pipeline":
        run_full_pipeline()
    elif args.command == "predict":
        predict_only()
    elif args.command == "auto":
        run_auto(args.interval)
    else:
        parser.print_help()
