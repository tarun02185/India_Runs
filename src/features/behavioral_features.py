"""
Behavioral feature extraction from Redrob signals.

Behavioral signals serve as a MODIFIER on base relevance (JD + signals doc).
They indicate whether a technically qualified candidate is actually reachable,
responsive, and actively in the job market.
"""

import logging
import math
from datetime import datetime
from typing import Any

from dateutil.parser import parse as parse_date

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]

# Reference date for computing recency
# Using a fixed reference to ensure reproducibility
REFERENCE_DATE = datetime(2026, 6, 1)


def compute_engagement_score(candidate: CandidateRecord) -> float:
    """
    Composite engagement score based on platform activity signals.

    Combines:
    - Recruiter response rate (high weight)
    - Response time (moderate weight)
    - Recency of last activity (moderate weight)
    - Open to work flag (low weight)

    Returns:
        Float [0, 1] where 1.0 = highly engaged, responsive candidate.
    """
    signals = candidate.get("redrob_signals", {})

    # Component 1: Recruiter response rate [0, 1]
    response_rate = signals.get("recruiter_response_rate", 0.5)
    if not isinstance(response_rate, (int, float)):
        response_rate = 0.5

    # Component 2: Response time — faster is better
    avg_response_hours = signals.get("avg_response_time_hours", 48)
    if not isinstance(avg_response_hours, (int, float)) or avg_response_hours < 0:
        avg_response_hours = 48  # assume average
    # Map to [0, 1]: <4h = 1.0, 24h = 0.7, 72h = 0.3, 168h+ = 0.1
    response_time_score = 1.0 / (1.0 + math.exp(0.03 * (avg_response_hours - 24)))

    # Component 3: Recency of last activity
    recency_score = _compute_recency_score(signals)

    # Component 4: Open to work flag
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.5

    # Weighted composite
    engagement = (
        0.40 * response_rate +
        0.20 * response_time_score +
        0.25 * recency_score +
        0.15 * open_to_work
    )

    return min(max(engagement, 0.0), 1.0)


def compute_platform_activity_score(candidate: CandidateRecord) -> float:
    """
    Score based on platform verification and profile completeness.

    Returns:
        Float [0, 1] measuring how "real" and complete the profile is.
    """
    signals = candidate.get("redrob_signals", {})

    # Profile completeness [0, 100] → [0, 1]
    completeness = signals.get("profile_completeness_score", 50)
    if not isinstance(completeness, (int, float)):
        completeness = 50
    completeness_norm = min(completeness / 100.0, 1.0)

    # Verification signals
    verified_email = 1.0 if signals.get("verified_email", False) else 0.0
    verified_phone = 1.0 if signals.get("verified_phone", False) else 0.0
    linkedin = 1.0 if signals.get("linkedin_connected", False) else 0.0

    # Interview completion rate
    interview_rate = signals.get("interview_completion_rate", -1)
    if not isinstance(interview_rate, (int, float)) or interview_rate < 0:
        interview_norm = 0.5  # neutral for missing
    else:
        interview_norm = interview_rate

    # Weighted composite
    activity = (
        0.30 * completeness_norm +
        0.15 * verified_email +
        0.15 * verified_phone +
        0.10 * linkedin +
        0.30 * interview_norm
    )

    return min(max(activity, 0.0), 1.0)


def compute_github_score(candidate: CandidateRecord) -> float:
    """
    Normalize GitHub activity score.

    -1 = no GitHub linked → neutral (0.5)
    0-100 → normalized [0, 1]

    Returns:
        Float [0, 1].
    """
    signals = candidate.get("redrob_signals", {})
    github = signals.get("github_activity_score", -1)

    if not isinstance(github, (int, float)) or github < 0:
        return 0.5  # neutral for missing

    return min(github / 100.0, 1.0)


def _compute_recency_score(signals: dict) -> float:
    """
    Score based on how recently the candidate was active.

    Active within 7 days = 1.0
    Active within 30 days = 0.8
    Active within 90 days = 0.5
    Active >180 days ago = 0.1
    """
    try:
        last_active = parse_date(signals.get("last_active_date", ""))
        days_since = (REFERENCE_DATE - last_active.replace(tzinfo=None)).days

        if days_since < 0:
            days_since = 0  # future date, treat as very recent

        if days_since <= 7:
            return 1.0
        elif days_since <= 30:
            return 0.8
        elif days_since <= 60:
            return 0.6
        elif days_since <= 90:
            return 0.4
        elif days_since <= 180:
            return 0.2
        else:
            return 0.1
    except (ValueError, TypeError):
        return 0.5  # neutral for unparseable dates


def extract_behavioral_features(candidate: CandidateRecord) -> dict[str, Any]:
    """
    Extract all behavioral features from Redrob signals.

    Returns dict with:
    - engagement_score: float [0, 1]
    - platform_activity: float [0, 1]
    - github_score: float [0, 1]
    - behavioral_composite: float [0, 1] (weighted combination)
    """
    engagement = compute_engagement_score(candidate)
    platform = compute_platform_activity_score(candidate)
    github = compute_github_score(candidate)

    # Behavioral composite: engagement is most important
    composite = (
        0.55 * engagement +
        0.30 * platform +
        0.15 * github
    )

    return {
        "engagement_score": engagement,
        "platform_activity": platform,
        "github_score": github,
        "behavioral_composite": min(max(composite, 0.0), 1.0),
    }
