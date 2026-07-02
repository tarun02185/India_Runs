"""
Honeypot candidate detector.

Identifies candidates with subtly impossible profiles using rule-based
anomaly detection. Each rule checks for a specific logical impossibility
that a real candidate profile cannot have.

The ~80 honeypots in the dataset are forced to tier 0 in ground truth.
>10% honeypot rate in top-100 = automatic disqualification.
"""

import logging
from datetime import datetime
from typing import Any

from dateutil.parser import parse as parse_date

from src.config import HONEYPOT_SCORE_THRESHOLD

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]


def compute_honeypot_score(candidate: CandidateRecord) -> float:
    """
    Compute an anomaly score indicating how likely a candidate is a honeypot.

    Higher score = more likely honeypot. Each rule contributes independently.
    Candidates with score >= HONEYPOT_SCORE_THRESHOLD are flagged.

    Args:
        candidate: A single candidate record dict.

    Returns:
        Float anomaly score (0.0 = clean, higher = more suspicious).
    """
    score = 0.0
    cid = candidate.get("candidate_id", "UNKNOWN")
    reasons: list[str] = []

    # ── Rule 1: Salary min > max ─────────────────────────────────────────
    score += _check_salary_inversion(candidate, reasons)

    # ── Rule 2: Signup date after last active date ───────────────────────
    score += _check_signup_after_active(candidate, reasons)

    # ── Rule 3: Expert/advanced proficiency with 0 duration ──────────────
    score += _check_skill_proficiency_without_duration(candidate, reasons)

    # ── Rule 4: Experience years vs career history sum mismatch ───────────
    score += _check_experience_duration_mismatch(candidate, reasons)

    # ── Rule 5: Career entry date inconsistencies ────────────────────────
    score += _check_career_date_inconsistencies(candidate, reasons)

    # ── Rule 6: Too many high-proficiency skills with zero endorsements ──
    score += _check_mass_unendorsed_expert_skills(candidate, reasons)

    # ── Rule 7: Duration_months doesn't match start→end date math ────────
    score += _check_duration_vs_date_math(candidate, reasons)

    # ── Rule 8: Career description recycling within candidate ────────────
    score += _check_description_recycling(candidate, reasons)

    if score >= HONEYPOT_SCORE_THRESHOLD:
        logger.info(
            "HONEYPOT flagged: %s (score=%.1f) — %s",
            cid, score, "; ".join(reasons),
        )

    return score


def detect_honeypots(
    candidates: list[CandidateRecord],
) -> dict[str, float]:
    """
    Compute honeypot scores for all candidates.

    Args:
        candidates: List of candidate records.

    Returns:
        Dict mapping candidate_id → honeypot anomaly score.
    """
    scores: dict[str, float] = {}
    flagged_count = 0

    for candidate in candidates:
        cid = candidate.get("candidate_id", "UNKNOWN")
        score = compute_honeypot_score(candidate)
        scores[cid] = score
        if score >= HONEYPOT_SCORE_THRESHOLD:
            flagged_count += 1

    logger.info(
        "Honeypot detection complete: %d/%d flagged (%.1f%%)",
        flagged_count,
        len(candidates),
        100.0 * flagged_count / len(candidates) if candidates else 0,
    )
    return scores


def is_honeypot(score: float) -> bool:
    """Check if a honeypot score indicates a flagged candidate."""
    return score >= HONEYPOT_SCORE_THRESHOLD


# ─── Individual Rule Implementations ─────────────────────────────────────────


def _check_salary_inversion(
    candidate: CandidateRecord, reasons: list[str],
) -> float:
    """Rule 1: salary min > max is logically impossible."""
    try:
        salary = candidate["redrob_signals"]["expected_salary_range_inr_lpa"]
        sal_min = salary.get("min", 0)
        sal_max = salary.get("max", 0)
        if isinstance(sal_min, (int, float)) and isinstance(sal_max, (int, float)):
            if sal_min > sal_max and sal_max > 0:
                reasons.append(f"salary min ({sal_min}) > max ({sal_max})")
                return 3.0
    except (KeyError, TypeError):
        pass
    return 0.0


def _check_signup_after_active(
    candidate: CandidateRecord, reasons: list[str],
) -> float:
    """Rule 2: signup_date after last_active_date is temporally impossible."""
    try:
        signals = candidate["redrob_signals"]
        signup = parse_date(signals["signup_date"])
        last_active = parse_date(signals["last_active_date"])
        if signup > last_active:
            reasons.append(
                f"signup ({signals['signup_date']}) after "
                f"last_active ({signals['last_active_date']})"
            )
            return 2.0
    except (KeyError, TypeError, ValueError):
        pass
    return 0.0


def _check_skill_proficiency_without_duration(
    candidate: CandidateRecord, reasons: list[str],
) -> float:
    """Rule 3: expert/advanced skills with 0 months of usage."""
    penalty = 0.0
    zero_duration_high_prof = 0
    skills = candidate.get("skills", [])
    for skill in skills:
        prof = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0)
        if prof in ("expert", "advanced") and duration == 0:
            zero_duration_high_prof += 1

    if zero_duration_high_prof >= 3:
        penalty = min(zero_duration_high_prof * 0.5, 3.0)
        reasons.append(
            f"{zero_duration_high_prof} advanced/expert skills with 0 months duration"
        )
    return penalty


def _check_experience_duration_mismatch(
    candidate: CandidateRecord, reasons: list[str],
) -> float:
    """Rule 4: years_of_experience doesn't match career history sum."""
    try:
        claimed_years = candidate["profile"]["years_of_experience"]
        career = candidate.get("career_history", [])
        if not career:
            return 0.0

        total_months = sum(entry.get("duration_months", 0) for entry in career)
        claimed_months = claimed_years * 12
        diff = abs(claimed_months - total_months)

        # Allow up to 36 months of gap (career gaps, overlaps, rounding)
        if diff > 48:
            reasons.append(
                f"claimed {claimed_years}yr ({claimed_months:.0f}mo) "
                f"but career sums to {total_months}mo (diff={diff:.0f}mo)"
            )
            return 2.0
        elif diff > 36:
            reasons.append(
                f"experience gap: claimed={claimed_months:.0f}mo, "
                f"actual={total_months}mo"
            )
            return 1.0
    except (KeyError, TypeError):
        pass
    return 0.0


def _check_career_date_inconsistencies(
    candidate: CandidateRecord, reasons: list[str],
) -> float:
    """Rule 5: start_date > end_date in career entries."""
    penalty = 0.0
    for entry in candidate.get("career_history", []):
        try:
            start = entry.get("start_date")
            end = entry.get("end_date")
            if start and end:
                start_dt = parse_date(start)
                end_dt = parse_date(end)
                if start_dt > end_dt:
                    reasons.append(
                        f"career start ({start}) after end ({end}) "
                        f"at {entry.get('company', '?')}"
                    )
                    penalty += 3.0
        except (ValueError, TypeError):
            pass
    return min(penalty, 6.0)


def _check_mass_unendorsed_expert_skills(
    candidate: CandidateRecord, reasons: list[str],
) -> float:
    """Rule 6: many high-proficiency skills with zero endorsements."""
    high_prof_no_endorse = 0
    skills = candidate.get("skills", [])
    for skill in skills:
        prof = skill.get("proficiency", "")
        endorsements = skill.get("endorsements", 0)
        if prof in ("expert", "advanced") and endorsements == 0:
            high_prof_no_endorse += 1

    if high_prof_no_endorse >= 5:
        reasons.append(
            f"{high_prof_no_endorse} advanced/expert skills with 0 endorsements"
        )
        return 2.0
    return 0.0


def _check_duration_vs_date_math(
    candidate: CandidateRecord, reasons: list[str],
) -> float:
    """Rule 7: duration_months doesn't match start_date→end_date gap."""
    penalty = 0.0
    for entry in candidate.get("career_history", []):
        try:
            start = entry.get("start_date")
            end = entry.get("end_date")
            stated_duration = entry.get("duration_months", 0)

            if start and end and stated_duration > 0:
                start_dt = parse_date(start)
                end_dt = parse_date(end)
                # Compute expected months between dates
                expected_months = (
                    (end_dt.year - start_dt.year) * 12
                    + (end_dt.month - start_dt.month)
                )
                diff = abs(expected_months - stated_duration)
                if diff > 12:
                    reasons.append(
                        f"duration mismatch at {entry.get('company', '?')}: "
                        f"stated={stated_duration}mo, "
                        f"computed={expected_months}mo"
                    )
                    penalty += 1.5
        except (ValueError, TypeError):
            pass
    return min(penalty, 4.5)


def _check_description_recycling(
    candidate: CandidateRecord, reasons: list[str],
) -> float:
    """Rule 8: same description text reused across career entries."""
    career = candidate.get("career_history", [])
    if len(career) <= 1:
        return 0.0

    descriptions = [
        entry.get("description", "").strip()
        for entry in career
        if entry.get("description", "").strip()
    ]
    if not descriptions:
        return 0.0

    unique = set(descriptions)
    if len(unique) < len(descriptions):
        recycled = len(descriptions) - len(unique)
        reasons.append(f"{recycled} recycled career descriptions")
        return 1.5

    return 0.0
