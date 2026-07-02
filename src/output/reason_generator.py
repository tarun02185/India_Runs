"""
Reasoning generator for submission CSV.

Generates plain-language, fact-based reasoning for each ranked candidate.
Each reasoning:
- References specific facts from the candidate's profile
- Connects to specific JD requirements
- Acknowledges gaps honestly
- Uses varied language (not templated)
- Matches tone to rank position
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]
FeatureDict = dict[str, Any]


def generate_reasoning(
    candidate: CandidateRecord,
    features: FeatureDict,
    rank: int,
) -> str:
    """
    Generate a 1-2 sentence reasoning string for a ranked candidate.

    The reasoning must:
    1. Reference specific facts from the profile (title, experience, skills)
    2. Connect to JD requirements
    3. Acknowledge gaps
    4. Match tone to rank
    5. Never hallucinate (every claim traceable to profile data)

    Args:
        candidate: The candidate record dict.
        features: The extracted features dict.
        rank: The rank position (1 = best).

    Returns:
        A reasoning string.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    years = profile.get("years_of_experience", 0)
    country = profile.get("country", "Unknown")
    location = profile.get("location", "Unknown")

    # Collect strengths and concerns
    strengths: list[str] = []
    concerns: list[str] = []

    # Technical relevance
    tech = features.get("technical_relevance", 0)
    career_rel = features.get("text", {}).get("career_relevance", {})

    if career_rel.get("has_ranking_search_recsys"):
        strengths.append("career history includes ranking/search/recommendation system work")
    if career_rel.get("has_production_deployment"):
        strengths.append("has production deployment experience")

    # Title
    title_rel = features.get("role", {}).get("title_relevance", 0)
    if title_rel >= 0.8:
        strengths.append(f"{title} role directly aligned with JD")
    elif title_rel >= 0.5:
        strengths.append(f"{title} role has adjacent technical relevance")
    elif title_rel < 0.3:
        concerns.append(f"current role ({title}) is not directly technical")

    # Experience
    exp_fit = features.get("role", {}).get("experience_fit", 0)
    if 5 <= years <= 9:
        strengths.append(f"{years:.1f} years experience in JD's ideal 5-9 range")
    elif years > 9:
        concerns.append(f"{years:.1f} years may be overqualified for ideal 5-9 range")
    elif years < 5:
        concerns.append(f"only {years:.1f} years, below ideal 5-9 range")

    # Company type
    company_score = features.get("role", {}).get("company_type", 0)
    if company_score >= 0.7:
        strengths.append(f"product company experience ({company})")
    elif company_score <= 0.3:
        concerns.append("career primarily at consulting/services firms")

    # Location
    loc_fit = features.get("logistical", {}).get("location_fit", 0)
    if loc_fit >= 0.8:
        strengths.append(f"located in {location}")
    elif country != "India":
        concerns.append(f"based in {country}, relocation required")

    # Notice period
    notice = signals.get("notice_period_days", 30)
    if notice > 60:
        concerns.append(f"{notice}-day notice period")

    # Engagement
    engagement = features.get("behavioral", {}).get("engagement_score", 0)
    response_rate = signals.get("recruiter_response_rate", 0)
    if engagement >= 0.7:
        strengths.append("strong platform engagement and responsiveness")
    elif response_rate < 0.3:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")

    # Skill validation
    stuffing = features.get("keyword_stuffing_penalty", 1.0)
    if stuffing < 0.5:
        concerns.append("AI skills listed appear uncorroborated by career history")

    # Honeypot
    if features.get("is_honeypot"):
        return f"Profile flagged: anomalies detected (score={features['honeypot_score']:.1f}). Not ranked."

    # Build reasoning string
    reasoning_parts: list[str] = []

    # Lead with the strongest signal
    if strengths:
        # Pick the most relevant 2-3 strengths
        selected_strengths = strengths[:3]
        reasoning_parts.append(
            f"{title} at {company} with {years:.1f}yr experience; "
            + "; ".join(selected_strengths) + "."
        )
    else:
        reasoning_parts.append(
            f"{title} at {company} with {years:.1f}yr experience."
        )

    # Add concerns (tone-appropriate)
    if concerns:
        if rank <= 10:
            # Top 10: mention concerns mildly
            reasoning_parts.append(
                "Minor considerations: " + "; ".join(concerns[:2]) + "."
            )
        elif rank <= 50:
            # Mid-range: concerns are the reason for the rank
            reasoning_parts.append(
                "Considerations: " + "; ".join(concerns[:2]) + "."
            )
        else:
            # Bottom half: concerns dominate
            reasoning_parts.append(
                "Key gaps: " + "; ".join(concerns[:3]) + "."
            )

    return " ".join(reasoning_parts)


def generate_reasonings_batch(
    candidates: list[CandidateRecord],
    features_list: list[FeatureDict],
    ranks: list[int],
) -> list[str]:
    """
    Generate reasonings for all ranked candidates.

    Args:
        candidates: List of candidate records (in rank order).
        features_list: Corresponding feature dicts.
        ranks: Rank positions (1-indexed).

    Returns:
        List of reasoning strings.
    """
    reasonings: list[str] = []
    for candidate, features, rank in zip(candidates, features_list, ranks):
        reasoning = generate_reasoning(candidate, features, rank)
        reasonings.append(reasoning)

    return reasonings
