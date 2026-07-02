"""
Role and career structure feature extraction.

Scores title relevance, career trajectory, company type (product vs services),
and experience level fit against JD requirements.
"""

import logging
from typing import Any

from src.config import (
    CONSULTING_FIRMS,
    NON_TECHNICAL_TITLES,
    STRONG_TECHNICAL_TITLES,
    MODERATE_TECHNICAL_TITLES,
)
from src.jd_requirements import JD

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]


def compute_title_relevance(candidate: CandidateRecord) -> float:
    """
    Score how relevant the candidate's current and historical titles are.

    Returns:
        Float score [0, 1] where:
        - 1.0 = AI/ML Engineer, Senior Data Scientist, etc.
        - 0.5-0.7 = Software Engineer, Backend Engineer, etc.
        - 0.0-0.2 = Marketing Manager, Accountant, etc.
    """
    # Current title
    current_title = candidate.get("profile", {}).get("current_title", "").lower().strip()
    career = candidate.get("career_history", [])
    all_titles = [entry.get("title", "").lower().strip() for entry in career]

    # Check current title against tiers
    current_score = _score_single_title(current_title)

    # Check historical titles — best historical title matters
    best_historical = 0.0
    for title in all_titles:
        best_historical = max(best_historical, _score_single_title(title))

    # Current title weighted more (0.6) than best historical (0.4)
    # because current title reflects where they ARE, not where they WERE
    return 0.6 * current_score + 0.4 * best_historical


def _score_single_title(title: str) -> float:
    """Score a single title string."""
    if not title:
        return 0.3  # neutral for missing

    # Check strong technical
    for st in STRONG_TECHNICAL_TITLES:
        if st in title or title in st:
            return 1.0

    # Check moderate technical
    for mt in MODERATE_TECHNICAL_TITLES:
        if mt in title or title in mt:
            return 0.6

    # Check non-technical
    for nt in NON_TECHNICAL_TITLES:
        if nt in title or title in nt:
            return 0.05

    # Default: unknown title — mildly below neutral
    return 0.3


def compute_experience_fit(candidate: CandidateRecord) -> float:
    """
    Score how well the candidate's experience level fits the JD.

    The JD says 5-9 years ideal, but "range, not requirement."
    We use a smooth bell curve centered on 6-8 years.

    Returns:
        Float [0, 1] where peak is around 7 years.
    """
    years = candidate.get("profile", {}).get("years_of_experience", 0)

    if years is None or years < 0:
        return 0.1

    # Hard boundaries
    if years < JD.experience_years_hard_min:
        return 0.05
    if years > JD.experience_years_hard_max:
        return 0.1

    # Ideal range: 5-9 years, peak at 7
    ideal_center = (JD.experience_years_ideal_min + JD.experience_years_ideal_max) / 2
    ideal_width = (JD.experience_years_ideal_max - JD.experience_years_ideal_min) / 2

    if JD.experience_years_ideal_min <= years <= JD.experience_years_ideal_max:
        return 1.0  # perfect range

    # Smooth dropoff outside ideal
    if years < JD.experience_years_ideal_min:
        distance = JD.experience_years_ideal_min - years
        return max(0.1, 1.0 - 0.15 * distance)
    else:
        distance = years - JD.experience_years_ideal_max
        return max(0.1, 1.0 - 0.08 * distance)


def compute_company_type_score(candidate: CandidateRecord) -> float:
    """
    Score based on company type: product vs consulting vs other.

    JD explicitly disfavors consulting-only careers (L67).
    Product company experience at software/tech companies is preferred.

    Returns:
        Float [0, 1] where:
        - 1.0 = career at product companies in software/tech
        - 0.3 = mixed
        - 0.1 = consulting-only career
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.3

    consulting_entries = 0
    product_entries = 0
    software_entries = 0

    for entry in career:
        company = entry.get("company", "").lower().strip()
        industry = entry.get("industry", "").lower().strip()

        # Check consulting firms
        is_consulting = any(
            firm in company for firm in CONSULTING_FIRMS
        )
        if is_consulting:
            consulting_entries += 1
        elif industry in ("software", "internet", "technology", "fintech",
                         "e-commerce", "food delivery", "saas"):
            software_entries += 1
            product_entries += 1
        elif industry not in ("it services", "consulting"):
            product_entries += 1

    total = len(career)
    if total == 0:
        return 0.3

    consulting_ratio = consulting_entries / total
    software_ratio = software_entries / total

    # Consulting-only career is a strong negative (JD L67)
    if consulting_ratio >= 1.0:
        return 0.1

    # Mostly consulting
    if consulting_ratio > 0.5:
        return 0.3

    # Software product company experience
    if software_ratio > 0.5:
        return 1.0
    elif software_ratio > 0.0:
        return 0.7

    # Non-consulting, non-software (manufacturing, etc.)
    return 0.4


def compute_tenure_stability(candidate: CandidateRecord) -> float:
    """
    Score tenure stability — the JD dislikes "title chasers" who switch
    every 1.5 years (L65). Prefers 3+ year commitment potential.

    Returns:
        Float [0, 1] where:
        - 1.0 = average tenure 3+ years
        - 0.5 = average tenure ~2 years
        - 0.2 = average tenure <1.5 years (title chaser pattern)
    """
    career = candidate.get("career_history", [])
    if len(career) <= 1:
        return 0.7  # can't judge with 1 entry

    # Calculate average tenure (in months) for completed roles
    completed_tenures = [
        entry.get("duration_months", 0)
        for entry in career
        if not entry.get("is_current", False) and entry.get("duration_months", 0) > 0
    ]

    if not completed_tenures:
        return 0.7

    avg_tenure_months = sum(completed_tenures) / len(completed_tenures)

    if avg_tenure_months >= 36:
        return 1.0
    elif avg_tenure_months >= 24:
        return 0.8
    elif avg_tenure_months >= 18:
        return 0.5
    elif avg_tenure_months >= 12:
        return 0.3
    else:
        return 0.1


def compute_career_recency(candidate: CandidateRecord) -> float:
    """
    Check if the candidate has recent technical roles (within last 3 years).

    JD says (L42): will not move forward if hasn't written code in 18 months.

    Returns:
        Float [0, 1] where:
        - 1.0 = current role is technical
        - 0.5 = last technical role was 1-3 years ago
        - 0.1 = no recent technical role
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.3

    # Check current role
    current = next((e for e in career if e.get("is_current", False)), None)
    if current:
        title_score = _score_single_title(current.get("title", "").lower())
        if title_score >= 0.5:
            return 1.0

    # Check most recent non-current role
    non_current = [e for e in career if not e.get("is_current", False)]
    if non_current:
        # Sort by end_date descending (most recent first)
        for entry in non_current:
            title_score = _score_single_title(entry.get("title", "").lower())
            if title_score >= 0.5:
                return 0.6
    return 0.2


def extract_role_features(candidate: CandidateRecord) -> dict[str, Any]:
    """
    Extract all role and career structure features.

    Returns dict with keys:
    - title_relevance: float [0, 1]
    - experience_fit: float [0, 1]
    - company_type: float [0, 1]
    - tenure_stability: float [0, 1]
    - career_recency: float [0, 1]
    - num_career_entries: int
    - total_career_months: int
    """
    career = candidate.get("career_history", [])
    total_months = sum(e.get("duration_months", 0) for e in career)

    return {
        "title_relevance": compute_title_relevance(candidate),
        "experience_fit": compute_experience_fit(candidate),
        "company_type": compute_company_type_score(candidate),
        "tenure_stability": compute_tenure_stability(candidate),
        "career_recency": compute_career_recency(candidate),
        "num_career_entries": len(career),
        "total_career_months": total_months,
    }
