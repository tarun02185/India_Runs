"""
Hybrid retriever with Reciprocal Rank Fusion (RRF).

Combines BM25 (lexical) and dense (semantic) retrieval results
using RRF to produce a unified candidate shortlist.

RRF formula: score(d) = sum_over_rankers( 1 / (k + rank_i(d)) )
where k is a constant (default 60, from the original RRF paper).
"""

import logging
from typing import Any

from src.config import RRF_K, HYBRID_RETRIEVAL_DEPTH

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    *result_lists: list[tuple[str, float]],
    k: int = RRF_K,
    top_k: int = HYBRID_RETRIEVAL_DEPTH,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each result list is a list of (candidate_id, score) tuples,
    already sorted by score descending.

    Args:
        *result_lists: One or more ranked result lists.
        k: RRF constant (default 60).
        top_k: Number of results to return after fusion.

    Returns:
        Fused list of (candidate_id, rrf_score) tuples, sorted descending.
    """
    rrf_scores: dict[str, float] = {}

    for results in result_lists:
        for rank_idx, (cid, _original_score) in enumerate(results):
            rank = rank_idx + 1  # 1-indexed
            rrf_contribution = 1.0 / (k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    logger.info(
        "RRF fusion: %d unique candidates from %d lists, returning top-%d",
        len(sorted_results), len(result_lists), top_k,
    )

    return sorted_results[:top_k]


def hybrid_retrieve(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    k: int = RRF_K,
    top_k: int = HYBRID_RETRIEVAL_DEPTH,
) -> list[tuple[str, float]]:
    """
    Combine BM25 and dense retrieval results using RRF.

    Args:
        bm25_results: BM25 retrieval results (cid, bm25_score).
        dense_results: Dense retrieval results (cid, similarity_score).
        k: RRF constant.
        top_k: Number of results to return.

    Returns:
        Fused results (cid, rrf_score), sorted descending.
    """
    # Log overlap statistics
    bm25_ids = set(cid for cid, _ in bm25_results)
    dense_ids = set(cid for cid, _ in dense_results)
    overlap = bm25_ids & dense_ids
    bm25_only = bm25_ids - dense_ids
    dense_only = dense_ids - bm25_ids

    logger.info(
        "Retrieval overlap: %d shared, %d BM25-only, %d dense-only",
        len(overlap), len(bm25_only), len(dense_only),
    )

    return reciprocal_rank_fusion(
        bm25_results, dense_results,
        k=k, top_k=top_k,
    )
