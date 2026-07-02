"""
Text feature extraction.

Detects templated summaries, analyzes career description relevance,
and extracts text-based signals for ranking.
"""

import re
import logging
from typing import Any

from src.config import TEMPLATE_SIGNATURES
from src.jd_requirements import JD

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]


def is_templated_summary(candidate: CandidateRecord) -> bool:
    """
    Check if the candidate's summary matches a known filler template.

    The dataset uses specific template phrases for non-technical candidates.
    Detection is via substring matching — the templates are regular enough
    that no model is needed.
    """
    summary = candidate.get("profile", {}).get("summary", "")
    if not summary:
        return False

    summary_lower = summary.lower()
    for sig in TEMPLATE_SIGNATURES:
        if sig.lower() in summary_lower:
            return True
    return False


def has_technical_summary(candidate: CandidateRecord) -> bool:
    """
    Check if the candidate's summary mentions specific technologies or systems.

    A technical summary contains concrete tools, frameworks, or system descriptions
    rather than generic management language.
    """
    summary = candidate.get("profile", {}).get("summary", "")
    if not summary:
        return False

    summary_lower = summary.lower()

    # Check for technical vocabulary — at least 2 matches indicates technical content
    tech_terms = [
        "python", "java", "sql", "spark", "airflow", "pytorch", "tensorflow",
        "kubernetes", "docker", "aws", "gcp", "azure", "api", "backend",
        "frontend", "database", "pipeline", "model", "algorithm",
        "machine learning", "deep learning", "nlp", "transformer",
        "embeddings", "retrieval", "ranking", "recommendation",
        "data engineering", "ml", "neural", "training", "inference",
        "microservice", "distributed", "kafka", "redis", "elasticsearch",
    ]

    matches = sum(1 for term in tech_terms if term in summary_lower)
    return matches >= 2


def compute_career_description_relevance(candidate: CandidateRecord) -> dict[str, float]:
    """
    Score career descriptions against JD requirements.

    Returns a dict with:
    - strong_match_count: number of strong positive signals found
    - moderate_match_count: number of moderate positive signals found
    - negative_match_count: number of negative signals found
    - relevance_score: normalized [0, 1] score
    - has_production_deployment: bool flag
    - has_ranking_search_recsys: bool flag
    """
    career = candidate.get("career_history", [])
    all_descriptions = " ".join(
        entry.get("description", "") for entry in career
    ).lower()

    # Also include summary
    summary = candidate.get("profile", {}).get("summary", "").lower()
    full_text = all_descriptions + " " + summary

    # Count strong positive matches
    strong_count = 0
    for signal in JD.career_strong_positive:
        if signal.lower() in full_text:
            strong_count += 1

    # Count moderate positive matches
    moderate_count = 0
    for signal in JD.career_moderate_positive:
        if signal.lower() in full_text:
            moderate_count += 1

    # Count negative matches
    negative_count = 0
    for signal in JD.career_negative:
        if signal.lower() in full_text:
            negative_count += 1

    # Specific flags
    has_production = any(
        term in full_text
        for term in ["production", "deployed", "shipped", "real users", "at scale"]
    )

    has_ranking_search_recsys = any(
        term in full_text
        for term in [
            "ranking", "search system", "search engine",
            "recommendation", "retrieval", "information retrieval",
            "what users see", "relevance",
        ]
    )

    # Compute normalized score
    # Strong matches are worth 3x moderate matches
    # Negative matches subtract
    raw_score = (strong_count * 3 + moderate_count * 1 - negative_count * 2)
    # Normalize to [0, 1] using a sigmoid-like mapping
    # A score of 10+ maps to ~0.9; a score of 0 maps to 0.5; negative maps to <0.5
    import math
    relevance_score = 1.0 / (1.0 + math.exp(-0.3 * raw_score))

    return {
        "strong_match_count": strong_count,
        "moderate_match_count": moderate_count,
        "negative_match_count": negative_count,
        "relevance_score": relevance_score,
        "has_production_deployment": has_production,
        "has_ranking_search_recsys": has_ranking_search_recsys,
    }


def get_concatenated_text(candidate: CandidateRecord) -> str:
    """
    Build a single text string from candidate's summary + career descriptions.

    This is the text that gets embedded and indexed for retrieval.
    """
    parts: list[str] = []

    # Summary
    summary = candidate.get("profile", {}).get("summary", "")
    if summary:
        parts.append(summary)

    # Headline
    headline = candidate.get("profile", {}).get("headline", "")
    if headline:
        parts.append(headline)

    # Career descriptions (most recent first)
    for entry in candidate.get("career_history", []):
        desc = entry.get("description", "")
        title = entry.get("title", "")
        if desc:
            parts.append(f"{title}: {desc}" if title else desc)

    return " ".join(parts)


def extract_text_features(candidate: CandidateRecord) -> dict[str, Any]:
    """
    Extract all text-based features for a candidate.

    Returns a dict with keys:
    - is_templated_summary: bool
    - has_technical_summary: bool
    - career_relevance: dict (from compute_career_description_relevance)
    - concatenated_text_length: int
    """
    return {
        "is_templated_summary": is_templated_summary(candidate),
        "has_technical_summary": has_technical_summary(candidate),
        "career_relevance": compute_career_description_relevance(candidate),
        "concatenated_text_length": len(get_concatenated_text(candidate)),
    }
