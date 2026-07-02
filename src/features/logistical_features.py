"""
Logistical feature extraction.

Scores location fit, notice period, salary range, and work mode
against JD requirements. Used as threshold-based modifiers per
the assumption audit: binary/threshold signals, not continuous scoring.
"""

import logging
from typing import Any

from src.jd_requirements import JD

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]


def compute_location_fit(candidate: CandidateRecord) -> float:
    """
    Score location compatibility with JD requirements.

    JD L74: Pune/Noida preferred, but flexible.
    JD L92: Hyderabad, Mumbai, Delhi NCR also welcome.
    Outside India: case-by-case, no visa sponsorship.

    Returns:
        Float [0, 1] where:
        - 1.0 = preferred location (Pune, Noida)
        - 0.8 = other acceptable Indian cities
        - 0.5 = India, other city, willing to relocate
        - 0.3 = India, not willing to relocate
        - 0.1 = Outside India, not willing to relocate
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    country = (profile.get("country", "") or "").lower().strip()
    location = (profile.get("location", "") or "").lower().strip()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # Check if in India
    is_india = country == "india"

    if not is_india:
        return 0.3 if willing_to_relocate else 0.1

    # Check preferred locations
    for pref in JD.location_preferences:
        if pref in location:
            return 1.0

    # In India but not preferred city
    if willing_to_relocate:
        return 0.7
    else:
        return 0.5


def compute_notice_period_score(candidate: CandidateRecord) -> float:
    """
    Score notice period fit.

    JD L75: sub-30-day ideal, can buy out 30 days, 30+ still in scope.

    Returns:
        Float [0, 1].
    """
    signals = candidate.get("redrob_signals", {})
    notice = signals.get("notice_period_days", 30)

    if not isinstance(notice, (int, float)):
        return 0.5

    if notice <= JD.notice_period_ideal_max:
        return 1.0
    elif notice <= JD.notice_period_acceptable_max:
        return 0.7
    elif notice <= JD.notice_period_concern:
        return 0.4
    else:
        return 0.2


def compute_work_mode_fit(candidate: CandidateRecord) -> float:
    """
    Score work mode compatibility.

    JD L74: Hybrid — flexible cadence. Remote is acceptable per industry norms.

    Returns:
        Float [0, 1].
    """
    signals = candidate.get("redrob_signals", {})
    work_mode = (signals.get("preferred_work_mode", "") or "").lower().strip()

    if work_mode in ("hybrid", "flexible"):
        return 1.0
    elif work_mode == "onsite":
        return 0.9  # fine for a hybrid role
    elif work_mode == "remote":
        return 0.7  # acceptable but not ideal for quarterly offsites
    else:
        return 0.5  # unknown


def compute_salary_reasonableness(candidate: CandidateRecord) -> float:
    """
    Basic salary reasonableness check.

    We don't have the company's salary range explicitly, but for a Senior
    AI Engineer at a Series A in India, ~25-50 LPA is plausible.

    This is a soft signal — salary expectations alone shouldn't disqualify.

    Returns:
        Float [0, 1].
    """
    signals = candidate.get("redrob_signals", {})
    salary = signals.get("expected_salary_range_inr_lpa", {})

    if not salary or not isinstance(salary, dict):
        return 0.5  # neutral for missing

    sal_min = salary.get("min", 0)
    sal_max = salary.get("max", 0)

    if not isinstance(sal_min, (int, float)) or not isinstance(sal_max, (int, float)):
        return 0.5

    # Use the midpoint for scoring
    if sal_max <= 0:
        return 0.5

    midpoint = (sal_min + sal_max) / 2

    # Reasonable range for Senior AI Engineer at Series A India
    if 15 <= midpoint <= 60:
        return 1.0
    elif 10 <= midpoint <= 80:
        return 0.7
    elif midpoint < 10:
        return 0.5  # unusually low, maybe junior?
    else:
        return 0.3  # very high expectations


def extract_logistical_features(candidate: CandidateRecord) -> dict[str, Any]:
    """
    Extract all logistical features.

    Returns dict with:
    - location_fit: float [0, 1]
    - notice_period: float [0, 1]
    - work_mode_fit: float [0, 1]
    - salary_reasonableness: float [0, 1]
    - logistical_composite: float [0, 1]
    """
    location = compute_location_fit(candidate)
    notice = compute_notice_period_score(candidate)
    work_mode = compute_work_mode_fit(candidate)
    salary = compute_salary_reasonableness(candidate)

    # Weighted composite — location is most important logistically
    composite = (
        0.40 * location +
        0.30 * notice +
        0.15 * work_mode +
        0.15 * salary
    )

    return {
        "location_fit": location,
        "notice_period": notice,
        "work_mode_fit": work_mode,
        "salary_reasonableness": salary,
        "logistical_composite": min(max(composite, 0.0), 1.0),
    }
