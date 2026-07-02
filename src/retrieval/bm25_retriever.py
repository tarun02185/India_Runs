"""
BM25 retriever for candidate search.

Uses the rank_bm25 library to perform lexical retrieval over candidate
career descriptions + summaries. The BM25 query is built from the
structured JDRequirements object.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config import BM25_RETRIEVAL_DEPTH, BM25_INDEX_PATH, CANDIDATE_IDS_PATH
from src.jd_requirements import JD
from src.features.text_features import get_concatenated_text

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.

    Strips punctuation and lowercases. No stemming — we want exact
    term matches for domain-specific vocabulary.
    """
    import re
    text = text.lower()
    # Keep hyphens and dots within words (e.g., "sentence-transformers", "A/B")
    tokens = re.findall(r"[a-z0-9][\w\-./]*[a-z0-9]|[a-z0-9]", text)
    return tokens


def build_bm25_index(
    candidates: list[CandidateRecord],
    save_path: Path | None = None,
) -> tuple[BM25Okapi, list[str]]:
    """
    Build a BM25 index over candidate texts.

    Args:
        candidates: List of candidate records.
        save_path: If set, save the index to disk for reuse.

    Returns:
        Tuple of (BM25 index, list of candidate_ids in index order).
    """
    logger.info("Building BM25 index for %d candidates...", len(candidates))
    start = time.time()

    corpus: list[list[str]] = []
    candidate_ids: list[str] = []

    for candidate in candidates:
        text = get_concatenated_text(candidate)
        tokens = tokenize(text)
        corpus.append(tokens)
        candidate_ids.append(candidate.get("candidate_id", "UNKNOWN"))

    bm25 = BM25Okapi(corpus)

    elapsed = time.time() - start
    logger.info("BM25 index built in %.2fs (%d documents)", elapsed, len(corpus))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(bm25, f)
        ids_path = save_path.parent / "candidate_ids.pkl"
        with open(ids_path, "wb") as f:
            pickle.dump(candidate_ids, f)
        logger.info("BM25 index saved to %s", save_path)

    return bm25, candidate_ids


def load_bm25_index(
    index_path: Path = BM25_INDEX_PATH,
    ids_path: Path = CANDIDATE_IDS_PATH,
) -> tuple[BM25Okapi, list[str]]:
    """Load a pre-saved BM25 index from disk."""
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(ids_path, "rb") as f:
        candidate_ids = pickle.load(f)
    logger.info("BM25 index loaded: %d documents", len(candidate_ids))
    return bm25, candidate_ids


def build_bm25_query() -> list[str]:
    """
    Build the BM25 query from structured JD requirements.

    Uses primary + secondary query terms from JDRequirements,
    ensuring the query covers both exact JD vocabulary and
    plain-language Tier 5 vocabulary.
    """
    query_text = " ".join(JD.retrieval_query_primary + JD.retrieval_query_secondary)
    tokens = tokenize(query_text)
    logger.info("BM25 query built: %d tokens", len(tokens))
    return tokens


def retrieve_bm25(
    bm25: BM25Okapi,
    candidate_ids: list[str],
    query_tokens: list[str] | None = None,
    top_k: int = BM25_RETRIEVAL_DEPTH,
) -> list[tuple[str, float]]:
    """
    Retrieve top-K candidates using BM25.

    Args:
        bm25: The BM25 index.
        candidate_ids: List of candidate_ids in index order.
        query_tokens: Tokenized query. If None, builds from JD requirements.
        top_k: Number of candidates to retrieve.

    Returns:
        List of (candidate_id, bm25_score) tuples, sorted by score descending.
    """
    if query_tokens is None:
        query_tokens = build_bm25_query()

    start = time.time()

    # Get scores for all documents
    scores = bm25.get_scores(query_tokens)

    # Get top-K indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [
        (candidate_ids[idx], float(scores[idx]))
        for idx in top_indices
        if scores[idx] > 0  # Only include candidates with non-zero score
    ]

    elapsed = time.time() - start
    logger.info(
        "BM25 retrieval: %d results in %.3fs (query: %d tokens)",
        len(results), elapsed, len(query_tokens),
    )

    return results
