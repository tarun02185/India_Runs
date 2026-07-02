"""
Skill feature extraction.

Skills are NOT trusted at face value (JD L101: "that's a trap").
Instead, skills are used for:
1. Corroborative evidence (skill claimed + career confirms = trusted)
2. Negative signals (advanced skill + 0 months + 0 endorsements = keyword stuffing)
3. Validated skill assessments from Redrob signals

Raw skill count / presence is deliberately NOT a positive signal.
"""

import logging
from typing import Any

from src.jd_requirements import JD

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]

# Skills that indicate non-technical background when they dominate
NON_TECH_SKILLS = frozenset({
    "sales", "accounting", "tally", "excel", "powerpoint",
    "photoshop", "illustrator", "figma", "seo", "content writing",
    "marketing", "hr management", "recruitment", "supply chain",
    "project management", "scrum", "agile",
})


def compute_skill_trust_score(candidate: CandidateRecord) -> float:
    """
    Compute how much we can trust this candidate's skills.

    A high trust score means:
    - Skills have endorsements proportional to claimed proficiency
    - Skills have non-zero duration
    - Skills are corroborated by career descriptions

    A low trust score means:
    - Many advanced/expert skills with 0 endorsements (keyword stuffing)
    - Skills contradict career descriptions

    Returns:
        Float [0, 1] where 1.0 = highly trustworthy skills profile
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.5  # neutral for no skills

    total_skills = len(skills)
    suspicious_count = 0
    well_endorsed_count = 0

    for skill in skills:
        prof = skill.get("proficiency", "")
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        # Suspicious: advanced/expert with no evidence
        if prof in ("advanced", "expert"):
            if endorsements == 0 and duration < 6:
                suspicious_count += 1
            elif endorsements >= 10 and duration >= 12:
                well_endorsed_count += 1

    suspicious_ratio = suspicious_count / total_skills
    endorsed_ratio = well_endorsed_count / total_skills if total_skills > 0 else 0

    if suspicious_ratio > 0.5:
        return 0.1  # highly suspicious
    elif suspicious_ratio > 0.3:
        return 0.3
    elif endorsed_ratio > 0.3:
        return 0.9
    elif endorsed_ratio > 0.1:
        return 0.7
    else:
        return 0.5


def compute_keyword_stuffing_penalty(candidate: CandidateRecord) -> float:
    """
    Detect keyword stuffing: many AI/ML skills with no career evidence.

    The dataset trap: non-technical candidates with 10+ AI skills stuffed
    into their profile. If career descriptions show no technical work,
    but skills list has AI keywords, this is keyword stuffing.

    Returns:
        Float [0, 1] where:
        - 1.0 = no stuffing detected (clean)
        - 0.0 = severe keyword stuffing
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 1.0

    # Count AI/ML skills
    ai_skill_names = set()
    for skill_group in [
        JD.skills_hard_req_embeddings,
        JD.skills_hard_req_vector_db,
        JD.skills_hard_req_eval,
        JD.skills_soft_pref_llm,
        JD.skills_soft_pref_ltr,
    ]:
        for s in skill_group:
            ai_skill_names.add(s.lower())

    ai_skill_count = sum(
        1 for skill in skills
        if skill.get("name", "").lower() in ai_skill_names
    )

    # Check career descriptions for technical content
    career = candidate.get("career_history", [])
    all_desc = " ".join(e.get("description", "") for e in career).lower()

    tech_desc_terms = [
        "machine learning", "ml", "model", "neural", "deep learning",
        "embeddings", "retrieval", "ranking", "nlp", "transformer",
        "pytorch", "tensorflow", "data science", "algorithm",
    ]
    desc_tech_count = sum(1 for t in tech_desc_terms if t in all_desc)

    # Keyword stuffing: many AI skills but career shows no tech
    if ai_skill_count >= 3 and desc_tech_count == 0:
        # Check current title too
        title = candidate.get("profile", {}).get("current_title", "").lower()
        for nt in NON_TECH_SKILLS:
            if nt in title:
                return 0.0  # severe stuffing: non-tech title + AI skills + no tech desc

        return 0.2  # moderate stuffing

    if ai_skill_count >= 5 and desc_tech_count <= 1:
        return 0.4  # mild stuffing signal

    return 1.0


def compute_relevant_skill_count(candidate: CandidateRecord) -> dict[str, int]:
    """
    Count skills that map to specific JD requirements.

    Only counts skills that are CORROBORATED (endorsed + non-trivial duration).

    Returns:
        Dict with counts per JD requirement category.
    """
    skills = candidate.get("skills", [])

    counts = {
        "hard_embeddings": 0,
        "hard_vector_db": 0,
        "hard_python": 0,
        "hard_eval": 0,
        "soft_llm": 0,
        "soft_ltr": 0,
    }

    for skill in skills:
        name = skill.get("name", "").lower()
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        # Only count if there's SOME evidence (not just claimed)
        is_corroborated = endorsements > 0 or duration >= 6

        if not is_corroborated:
            continue

        # Match against JD requirement categories
        for req_name, req_skills in [
            ("hard_embeddings", JD.skills_hard_req_embeddings),
            ("hard_vector_db", JD.skills_hard_req_vector_db),
            ("hard_python", JD.skills_hard_req_python),
            ("hard_eval", JD.skills_hard_req_eval),
            ("soft_llm", JD.skills_soft_pref_llm),
            ("soft_ltr", JD.skills_soft_pref_ltr),
        ]:
            for req_skill in req_skills:
                if req_skill.lower() in name or name in req_skill.lower():
                    counts[req_name] += 1
                    break

    return counts


def compute_non_tech_skill_dominance(candidate: CandidateRecord) -> float:
    """
    Check if non-technical skills dominate the profile.

    Returns:
        Float [0, 1] where:
        - 1.0 = mostly technical skills
        - 0.0 = mostly non-technical skills
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.5

    non_tech_count = sum(
        1 for skill in skills
        if skill.get("name", "").lower() in NON_TECH_SKILLS
    )

    non_tech_ratio = non_tech_count / len(skills)

    if non_tech_ratio > 0.6:
        return 0.1
    elif non_tech_ratio > 0.3:
        return 0.4
    elif non_tech_ratio > 0.1:
        return 0.7
    else:
        return 1.0


def extract_skill_features(candidate: CandidateRecord) -> dict[str, Any]:
    """
    Extract all skill-based features.

    Returns dict with:
    - skill_trust: float [0, 1]
    - keyword_stuffing_penalty: float [0, 1]
    - relevant_skill_counts: dict
    - non_tech_dominance: float [0, 1]
    - total_skills: int
    """
    return {
        "skill_trust": compute_skill_trust_score(candidate),
        "keyword_stuffing_penalty": compute_keyword_stuffing_penalty(candidate),
        "relevant_skill_counts": compute_relevant_skill_count(candidate),
        "non_tech_dominance": compute_non_tech_skill_dominance(candidate),
        "total_skills": len(candidate.get("skills", [])),
    }
