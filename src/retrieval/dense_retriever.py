"""
Dense retriever using sentence-transformers + FAISS.

Generates embeddings for candidate texts and performs approximate
nearest-neighbor search against a JD query embedding.

Pre-computation: Embeddings and FAISS index are built offline
(via scripts/precompute.py) and loaded at ranking time.
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Any

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    DENSE_RETRIEVAL_DEPTH,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    CANDIDATE_IDS_PATH,
    ARTIFACTS_DIR,
)
from src.jd_requirements import JD
from src.features.text_features import get_concatenated_text

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]


def build_jd_query_text() -> str:
    """
    Build a single text string representing the JD for dense retrieval.

    Combines primary and secondary query terms plus key JD phrases
    into a passage that captures what the ideal candidate looks like.
    """
    parts = [
        "Senior AI Engineer with production experience building ranking, "
        "retrieval, search, and recommendation systems.",
        "Strong Python, embeddings-based retrieval, vector databases, "
        "hybrid search infrastructure.",
        "Experience with NDCG, MRR, MAP evaluation frameworks, A/B testing.",
        "Shipped end-to-end ML systems to real users at product companies.",
        "Experience with sentence-transformers, FAISS, Pinecone, Weaviate, "
        "Elasticsearch, OpenSearch.",
        "LLM fine-tuning, learning-to-rank models, distributed systems.",
        " ".join(JD.retrieval_query_primary),
        " ".join(JD.retrieval_query_secondary),
    ]
    return " ".join(parts)


def generate_embeddings(
    candidates: list[CandidateRecord],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    save_path: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate embeddings for all candidate texts.

    Args:
        candidates: List of candidate records.
        model_name: sentence-transformers model name.
        batch_size: Batch size for encoding.
        save_path: If set, save embeddings to disk.

    Returns:
        Tuple of (embeddings array [N, dim], list of candidate_ids).
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    # Extract texts
    texts: list[str] = []
    candidate_ids: list[str] = []
    for c in candidates:
        texts.append(get_concatenated_text(c))
        candidate_ids.append(c.get("candidate_id", "UNKNOWN"))

    logger.info("Encoding %d texts (batch_size=%d)...", len(texts), batch_size)
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
    )

    elapsed = time.time() - start
    logger.info(
        "Encoding complete: %d embeddings in %.1fs (%.0f texts/sec)",
        len(embeddings), elapsed, len(texts) / elapsed if elapsed > 0 else 0,
    )

    embeddings = np.array(embeddings, dtype=np.float32)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, embeddings)
        logger.info("Embeddings saved to %s (%.1f MB)",
                     save_path, embeddings.nbytes / 1e6)

    return embeddings, candidate_ids


def build_faiss_index(
    embeddings: np.ndarray,
    save_path: Path | None = None,
) -> Any:
    """
    Build a FAISS index from embeddings.

    Uses IndexFlatIP (inner product = cosine similarity for L2-normalized vectors).

    Args:
        embeddings: Array of shape [N, dim].
        save_path: If set, save the index to disk.

    Returns:
        FAISS index.
    """
    import faiss

    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP (dim=%d, n=%d)", dim, len(embeddings))

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info("FAISS index built: %d vectors", index.ntotal)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_path))
        logger.info("FAISS index saved to %s", save_path)

    return index


def load_dense_artifacts(
    embeddings_path: Path = EMBEDDINGS_PATH,
    index_path: Path = FAISS_INDEX_PATH,
    ids_path: Path = CANDIDATE_IDS_PATH,
) -> tuple[Any, list[str]]:
    """
    Load pre-computed FAISS index and candidate IDs.

    Returns:
        Tuple of (FAISS index, list of candidate_ids).
    """
    import faiss
    import pickle

    index = faiss.read_index(str(index_path))
    logger.info("FAISS index loaded: %d vectors", index.ntotal)

    with open(ids_path, "rb") as f:
        candidate_ids = pickle.load(f)

    return index, candidate_ids


def retrieve_dense(
    index: Any,
    candidate_ids: list[str],
    query_text: str | None = None,
    model_name: str = EMBEDDING_MODEL_NAME,
    top_k: int = DENSE_RETRIEVAL_DEPTH,
) -> list[tuple[str, float]]:
    """
    Retrieve top-K candidates using dense retrieval.

    Args:
        index: FAISS index.
        candidate_ids: List of candidate_ids in index order.
        query_text: Query text. If None, builds from JD.
        model_name: Model used for encoding (must match index).
        top_k: Number of candidates to retrieve.

    Returns:
        List of (candidate_id, similarity_score) tuples, sorted descending.
    """
    from sentence_transformers import SentenceTransformer

    if query_text is None:
        query_text = build_jd_query_text()

    logger.info("Encoding query for dense retrieval...")
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
    ).astype(np.float32)

    start = time.time()
    scores, indices = index.search(query_embedding, min(top_k, index.ntotal))
    elapsed = time.time() - start

    results = [
        (candidate_ids[idx], float(scores[0][i]))
        for i, idx in enumerate(indices[0])
        if idx >= 0 and scores[0][i] > 0
    ]

    logger.info(
        "Dense retrieval: %d results in %.3fs",
        len(results), elapsed,
    )

    return results
