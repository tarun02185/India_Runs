"""
rank.py — Main entry point for candidate ranking.

This is the script that runs within the 5-minute CPU constraint.
It orchestrates the full pipeline:

  LOAD -> RETRIEVE (BM25 + Dense, RRF fusion) -> SCORE -> RANK -> REASON -> OUTPUT

Usage:
    python rank.py                           # Full pipeline on candidates.jsonl
    python rank.py --sample                  # Quick test on sample_candidates.json
    python rank.py --input path/to/data.jsonl
    python rank.py --output path/to/output.csv
    python rank.py --no-dense                # Skip dense retrieval (BM25 only)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    TOP_K,
    OUTPUT_DIR,
    HYBRID_RETRIEVAL_DEPTH,
    BM25_RETRIEVAL_DEPTH,
    DENSE_RETRIEVAL_DEPTH,
    FAISS_INDEX_PATH,
    CANDIDATE_IDS_PATH,
)
from src.data_loader import load_candidates, load_sample_candidates
from src.feature_pipeline import extract_all_features, extract_features_batch
from src.output.reason_generator import generate_reasonings_batch
from src.output.submission_generator import generate_submission, validate_submission


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranking System",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Run on sample_candidates.json (50 candidates, for testing)",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to candidates file (JSONL, JSONL.GZ, or JSON)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path for output CSV",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-bm25", action="store_true",
        help="Skip BM25 retrieval; score all candidates directly",
    )
    parser.add_argument(
        "--no-dense", action="store_true",
        help="Skip dense retrieval; use BM25 only",
    )
    return parser.parse_args()


def run_pipeline(
    input_path: Path | None = None,
    output_path: Path | None = None,
    use_sample: bool = False,
    use_bm25: bool = True,
    use_dense: bool = True,
) -> Path:
    """
    Run the full ranking pipeline.

    Args:
        input_path: Path to candidates file. If None, auto-detect.
        output_path: Path for output CSV.
        use_sample: If True, use sample_candidates.json.
        use_bm25: If True, use BM25 for retrieval.
        use_dense: If True, use dense retrieval (requires pre-computed artifacts).

    Returns:
        Path to the generated submission CSV.
    """
    pipeline_start = time.time()
    logger = logging.getLogger("pipeline")

    # ── Stage 1: LOAD ────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading candidates")
    logger.info("=" * 60)

    if use_sample:
        candidates = load_sample_candidates()
    else:
        candidates = load_candidates(input_path)

    total_candidates = len(candidates)
    logger.info("Loaded %d candidates in %.1fs", total_candidates, time.time() - stage_start)

    # Build lookup dict for candidate records
    candidate_lookup = {c["candidate_id"]: c for c in candidates}

    # ── Stage 2: RETRIEVE ────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("=" * 60)
    logger.info("STAGE 2: Retrieval")
    logger.info("=" * 60)

    retrieved_ids: set[str] | None = None

    if total_candidates <= HYBRID_RETRIEVAL_DEPTH:
        logger.info("Small dataset (%d <= %d) — skipping retrieval, scoring all",
                     total_candidates, HYBRID_RETRIEVAL_DEPTH)
    else:
        bm25_results: list[tuple[str, float]] = []
        dense_results: list[tuple[str, float]] = []

        # BM25 retrieval
        if use_bm25:
            from src.retrieval.bm25_retriever import (
                build_bm25_index,
                build_bm25_query,
                retrieve_bm25,
            )

            bm25_index, bm25_ids = build_bm25_index(candidates)
            query = build_bm25_query()
            bm25_results = retrieve_bm25(
                bm25_index, bm25_ids, query,
                top_k=BM25_RETRIEVAL_DEPTH,
            )
            logger.info("BM25: %d candidates retrieved", len(bm25_results))

        # Dense retrieval (only if pre-computed artifacts exist)
        if use_dense:
            if FAISS_INDEX_PATH.exists() and CANDIDATE_IDS_PATH.exists():
                from src.retrieval.dense_retriever import (
                    load_dense_artifacts,
                    retrieve_dense,
                )
                try:
                    faiss_index, dense_ids = load_dense_artifacts()
                    dense_results = retrieve_dense(
                        faiss_index, dense_ids,
                        top_k=DENSE_RETRIEVAL_DEPTH,
                    )
                    logger.info("Dense: %d candidates retrieved", len(dense_results))
                except Exception as e:
                    logger.warning("Dense retrieval failed, falling back to BM25 only: %s", e)
                    dense_results = []
            else:
                logger.info(
                    "Dense retrieval artifacts not found at %s — using BM25 only. "
                    "Run 'python scripts/precompute.py' to generate them.",
                    FAISS_INDEX_PATH,
                )

        # Fuse results
        if bm25_results and dense_results:
            from src.retrieval.hybrid_retriever import hybrid_retrieve
            fused = hybrid_retrieve(
                bm25_results, dense_results,
                top_k=HYBRID_RETRIEVAL_DEPTH,
            )
            retrieved_ids = set(cid for cid, _ in fused)
            logger.info("Hybrid (RRF): %d candidates after fusion", len(retrieved_ids))
        elif bm25_results:
            retrieved_ids = set(cid for cid, _ in bm25_results[:HYBRID_RETRIEVAL_DEPTH])
            logger.info("BM25 only: %d candidates", len(retrieved_ids))
        elif dense_results:
            retrieved_ids = set(cid for cid, _ in dense_results[:HYBRID_RETRIEVAL_DEPTH])
            logger.info("Dense only: %d candidates", len(retrieved_ids))
        else:
            logger.info("No retrieval performed — scoring all candidates")

    # Build the scoring set
    if retrieved_ids is not None:
        candidates_to_score = [
            c for c in candidates if c["candidate_id"] in retrieved_ids
        ]
    else:
        candidates_to_score = candidates

    logger.info("Candidates to score: %d (retrieval took %.1fs)",
                len(candidates_to_score), time.time() - stage_start)

    # ── Stage 3: FEATURE EXTRACTION & SCORING ────────────────────────────
    stage_start = time.time()
    logger.info("=" * 60)
    logger.info("STAGE 3: Feature Extraction & Scoring")
    logger.info("=" * 60)

    features_list = extract_features_batch(candidates_to_score, show_progress=True)

    logger.info("Feature extraction took %.1fs", time.time() - stage_start)

    # ── Stage 4: RANK ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4: Ranking")
    logger.info("=" * 60)

    # Sort by final_score descending, then candidate_id ascending (tie-break rule)
    features_list.sort(key=lambda f: (-f["final_score"], f["candidate_id"]))

    # Take top K
    top_k_features = features_list[:TOP_K]

    # Pad if BM25 didn't return enough and we need more
    if len(top_k_features) < TOP_K and total_candidates >= TOP_K:
        logger.warning(
            "Only %d candidates scored above 0. Padding to %d with remaining candidates.",
            len(top_k_features), TOP_K,
        )
        scored_ids = set(f["candidate_id"] for f in top_k_features)
        remaining = [c for c in candidates if c["candidate_id"] not in scored_ids]
        for c in remaining:
            if len(top_k_features) >= TOP_K:
                break
            top_k_features.append(extract_all_features(c))

    logger.info("Top-100 selected. Score range: %.4f — %.4f",
                top_k_features[0]["final_score"],
                top_k_features[-1]["final_score"] if top_k_features else 0)

    # ── Stage 5: REASONING ───────────────────────────────────────────────
    stage_start = time.time()
    logger.info("=" * 60)
    logger.info("STAGE 5: Generating Reasoning")
    logger.info("=" * 60)

    ranked_candidates = [
        candidate_lookup[f["candidate_id"]]
        for f in top_k_features
    ]
    ranks = list(range(1, len(top_k_features) + 1))
    reasonings = generate_reasonings_batch(ranked_candidates, top_k_features, ranks)

    logger.info("Reasoning generated in %.1fs", time.time() - stage_start)

    # ── Stage 6: OUTPUT ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 6: Generating Output")
    logger.info("=" * 60)

    if output_path is None:
        output_path = OUTPUT_DIR / "submission.csv"

    csv_path = generate_submission(top_k_features, reasonings, output_path)

    # Validate
    valid_ids = set(c["candidate_id"] for c in candidates)
    errors = validate_submission(csv_path, valid_ids)
    if errors:
        logger.error("SUBMISSION VALIDATION FAILED:")
        for err in errors:
            logger.error("  - %s", err)
    else:
        logger.info("SUBMISSION VALIDATION PASSED")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    hp_in_top100 = sum(1 for f in top_k_features if f["is_honeypot"])
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total time: %.2fs (budget: 300s, margin: %.1f%%)",
                elapsed, 100 * (300 - elapsed) / 300)
    logger.info("Candidates loaded: %d", total_candidates)
    logger.info("Candidates scored: %d", len(candidates_to_score))
    logger.info("Output: %s", csv_path)

    if top_k_features:
        logger.info(
            "Rank 1: %s (score=%.4f)",
            top_k_features[0]["candidate_id"],
            top_k_features[0]["final_score"],
        )
        # Log top-5 for quick inspection
        for i, f in enumerate(top_k_features[:5]):
            c = candidate_lookup[f["candidate_id"]]
            logger.info(
                "  #%d: %s — %s at %s (%.1fyr, %s) — score=%.4f",
                i + 1,
                f["candidate_id"],
                c["profile"].get("current_title", "?"),
                c["profile"].get("current_company", "?"),
                c["profile"].get("years_of_experience", 0),
                c["profile"].get("country", "?"),
                f["final_score"],
            )

    logger.info("Honeypots in top-100: %d (limit: 10)", hp_in_top100)
    if hp_in_top100 > 10:
        logger.error("HONEYPOT RATE EXCEEDS 10%% — DISQUALIFICATION RISK")

    return csv_path


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    try:
        csv_path = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            use_sample=args.sample,
            use_bm25=not args.no_bm25,
            use_dense=not args.no_dense,
        )
        print(f"\nSubmission written to: {csv_path}")
    except Exception as e:
        logging.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
