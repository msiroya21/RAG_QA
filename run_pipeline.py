import argparse
import logging

from src.config import ensure_dirs
from src.ingest.pipeline import run_pipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    args = parser.parse_args()

    ensure_dirs()
    run_pipeline()


if __name__ == "__main__":
    main()

