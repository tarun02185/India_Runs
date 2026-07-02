"""
Precompute script — generates embeddings and FAISS index offline.

This is run ONCE before ranking. The artifacts are saved to disk
and loaded by rank.py at runtime.

Usage:
    python scripts/precompute.py                    # Full 100K dataset
    python scripts/precompute.py --sample           # Sample (50 candidates)
    python scripts/precompute.py --input path.jsonl
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ARTIFACTS_DIR,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    CANDIDATE_IDS_PATH,
    BM25_INDEX_PATH,
)
from src.data_loader import load_candidates, load_sample_candidates
from src.retrieval.bm25_retriever import build_bm25_index
from src.retrieval.dense_retriever import (
    generate_embeddings,
    build_faiss_index,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute embeddings and indexes")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    parser.add_argument("--input", type=Path, default=None, help="Custom input file")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense embeddings")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("precompute")

    start = time.time()

    # Load candidates
    if args.sample:
        candidates = load_sample_candidates()
    else:
        candidates = load_candidates(args.input)

    logger.info("Loaded %d candidates", len(candidates))

    # Ensure artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build BM25 index
    if not args.skip_bm25:
        logger.info("=" * 50)
        logger.info("Building BM25 index...")
        logger.info("=" * 50)
        build_bm25_index(candidates, save_path=BM25_INDEX_PATH)

    # Generate dense embeddings + FAISS index
    if not args.skip_dense:
        logger.info("=" * 50)
        logger.info("Generating dense embeddings...")
        logger.info("=" * 50)
        embeddings, candidate_ids = generate_embeddings(
            candidates, save_path=EMBEDDINGS_PATH,
        )

        logger.info("=" * 50)
        logger.info("Building FAISS index...")
        logger.info("=" * 50)
        build_faiss_index(embeddings, save_path=FAISS_INDEX_PATH)

        # Save candidate IDs mapping
        with open(CANDIDATE_IDS_PATH, "wb") as f:
            pickle.dump(candidate_ids, f)
        logger.info("Candidate IDs saved to %s", CANDIDATE_IDS_PATH)

    elapsed = time.time() - start
    logger.info("=" * 50)
    logger.info("PRECOMPUTE COMPLETE in %.1fs", elapsed)
    logger.info("Artifacts in: %s", ARTIFACTS_DIR)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
