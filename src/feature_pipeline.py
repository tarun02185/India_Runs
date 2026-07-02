"""
Feature pipeline orchestrator.

Extracts ALL features for a candidate in a single pass,
producing a flat feature dict that the scorer consumes.
"""

import logging
import math
from typing import Any

from src.config import SIGMOID_GATE_STEEPNESS, SIGMOID_GATE_THRESHOLD, SCORING_WEIGHTS
from src.honeypot_detector import compute_honeypot_score, is_honeypot
from src.features.text_features import extract_text_features
from src.features.role_features import extract_role_features
from src.features.skill_features import extract_skill_features
from src.features.behavioral_features import extract_behavioral_features
from src.features.logistical_features import extract_logistical_features

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]
FeatureDict = dict[str, Any]


def extract_all_features(candidate: CandidateRecord) -> FeatureDict:
    """
    Extract all features for a single candidate.

    This is the single entry point for feature extraction. It calls
    all sub-extractors and combines their output into a flat dict.

    Args:
        candidate: A single candidate record dict.

    Returns:
        A flat dict containing all extracted features plus computed
        component scores and the final composite score.
    """
    cid = candidate.get("candidate_id", "UNKNOWN")

    # 1. Extract raw features from each module
    text_features = extract_text_features(candidate)
    role_features = extract_role_features(candidate)
    skill_features = extract_skill_features(candidate)
    behavioral_features = extract_behavioral_features(candidate)
    logistical_features = extract_logistical_features(candidate)

    # 2. Honeypot detection
    honeypot_score = compute_honeypot_score(candidate)
    is_hp = is_honeypot(honeypot_score)

    # 3. Compute component scores (each [0, 1])

    # Technical relevance: career descriptions + title + corroborated skills
    career_rel = text_features["career_relevance"]
    skill_counts = skill_features["relevant_skill_counts"]

    # How many of the 4 hard requirements are covered by corroborated skills?
    hard_req_coverage = min(
        1.0,
        (
            (1.0 if skill_counts["hard_embeddings"] > 0 else 0.0) +
            (1.0 if skill_counts["hard_vector_db"] > 0 else 0.0) +
            (1.0 if skill_counts["hard_python"] > 0 else 0.0) +
            (1.0 if skill_counts["hard_eval"] > 0 else 0.0)
        ) / 4.0
    )

    # Soft preference bonus
    soft_bonus = min(
        0.2,
        (
            (0.1 if skill_counts["soft_llm"] > 0 else 0.0) +
            (0.1 if skill_counts["soft_ltr"] > 0 else 0.0)
        )
    )

    technical_relevance = min(1.0, (
        0.35 * career_rel["relevance_score"] +
        0.25 * role_features["title_relevance"] +
        0.20 * hard_req_coverage +
        0.10 * skill_features["non_tech_dominance"] +
        0.10 * (1.0 if career_rel["has_ranking_search_recsys"] else 0.0) +
        soft_bonus
    ))

    # Penalize templated summaries
    if text_features["is_templated_summary"]:
        technical_relevance *= 0.3

    # Career structure score
    career_structure = (
        0.30 * role_features["company_type"] +
        0.25 * role_features["experience_fit"] +
        0.20 * role_features["tenure_stability"] +
        0.15 * role_features["career_recency"] +
        0.10 * min(role_features["num_career_entries"] / 4.0, 1.0)
    )

    # Behavioral composite (from module)
    behavioral = behavioral_features["behavioral_composite"]

    # Logistical composite (from module)
    logistical = logistical_features["logistical_composite"]

    # Misc signals
    misc = (
        0.50 * behavioral_features["github_score"] +
        0.50 * skill_features["skill_trust"]
    )

    # 4. Apply sigmoid gate on technical relevance
    sigmoid_multiplier = 1.0 / (
        1.0 + math.exp(
            -SIGMOID_GATE_STEEPNESS * (technical_relevance - SIGMOID_GATE_THRESHOLD)
        )
    )

    # 5. Compute weighted composite score
    weighted_sum = (
        SCORING_WEIGHTS.technical_relevance * technical_relevance +
        SCORING_WEIGHTS.career_structure * career_structure +
        SCORING_WEIGHTS.behavioral * behavioral +
        SCORING_WEIGHTS.logistical * logistical +
        SCORING_WEIGHTS.misc * misc
    )

    # 6. Apply modifiers
    # Sigmoid gate: smooth technical relevance threshold
    final_score = weighted_sum * sigmoid_multiplier

    # Keyword stuffing penalty
    final_score *= skill_features["keyword_stuffing_penalty"]

    # Honeypot penalty: zero out flagged candidates
    honeypot_penalty = 0.0 if is_hp else 1.0
    final_score *= honeypot_penalty

    # 7. Build result
    result: FeatureDict = {
        "candidate_id": cid,

        # Raw features (for debugging and reasoning)
        "text": text_features,
        "role": role_features,
        "skill": skill_features,
        "behavioral": behavioral_features,
        "logistical": logistical_features,

        # Component scores [0, 1]
        "technical_relevance": technical_relevance,
        "career_structure": career_structure,
        "behavioral_composite": behavioral,
        "logistical_composite": logistical,
        "misc_score": misc,

        # Modifiers
        "sigmoid_multiplier": sigmoid_multiplier,
        "keyword_stuffing_penalty": skill_features["keyword_stuffing_penalty"],
        "honeypot_score": honeypot_score,
        "is_honeypot": is_hp,
        "honeypot_penalty": honeypot_penalty,

        # Final score
        "final_score": final_score,
    }

    return result


def extract_features_batch(
    candidates: list[CandidateRecord],
    show_progress: bool = True,
) -> list[FeatureDict]:
    """
    Extract features for all candidates.

    Args:
        candidates: List of candidate records.
        show_progress: Whether to show progress logging.

    Returns:
        List of FeatureDict, one per candidate, in same order.
    """
    results: list[FeatureDict] = []
    total = len(candidates)

    for i, candidate in enumerate(candidates):
        features = extract_all_features(candidate)
        results.append(features)

        if show_progress and (i + 1) % 10000 == 0:
            logger.info("Features extracted: %d/%d (%.1f%%)", i + 1, total, 100 * (i + 1) / total)

    logger.info("Feature extraction complete: %d candidates", total)

    # Log summary statistics
    scores = [r["final_score"] for r in results]
    honeypots = sum(1 for r in results if r["is_honeypot"])
    templated = sum(1 for r in results if r["text"]["is_templated_summary"])

    logger.info(
        "Score stats: min=%.4f, max=%.4f, mean=%.4f, median=%.4f",
        min(scores), max(scores),
        sum(scores) / len(scores),
        sorted(scores)[len(scores) // 2],
    )
    logger.info("Honeypots flagged: %d (%.1f%%)", honeypots, 100 * honeypots / total if total else 0)
    logger.info("Templated summaries: %d (%.1f%%)", templated, 100 * templated / total if total else 0)

    return results
