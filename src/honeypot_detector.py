"""
honeypot_detector.py — Deterministic rules to identify and flag
impossible/trap candidate profiles in the Redrob dataset.

The dataset contains ~80 honeypot candidates with subtly impossible profiles.
If >10% of our top 100 are honeypots, we are disqualified.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Reference date for temporal calculations
REFERENCE_DATE = datetime(2026, 6, 24)


def detect_honeypot(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Run all honeypot detection rules against a candidate.

    Returns:
        (is_honeypot: bool, reason: str)
    """
    checks = [
        _check_job_duration_mismatch,
        _check_expert_skills_zero_duration,
        _check_experience_vs_career_mismatch,
    ]

    for check_fn in checks:
        is_bad, reason = check_fn(candidate)
        if is_bad:
            return True, reason

    return False, ""


def _check_job_duration_mismatch(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if any job's stated duration_months is wildly inconsistent
    with the actual time between start_date and end_date.

    Example honeypot: CAND_0008960 claims 171 months at a job
    that started only 21 months ago.
    """
    career = candidate.get("career_history", [])

    for job in career:
        stated_duration = job.get("duration_months", 0)
        start_date_str = job.get("start_date")
        end_date_str = job.get("end_date")

        if not start_date_str:
            continue

        try:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
            if end_date_str:
                end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
            else:
                end_dt = REFERENCE_DATE

            actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

            # Flag if difference is more than 6 months
            if abs(stated_duration - actual_months) > 6:
                return True, (
                    f"Job duration mismatch at {job.get('company', '?')}: "
                    f"stated {stated_duration}mo vs actual ~{actual_months}mo"
                )
        except (ValueError, TypeError):
            continue

    return False, ""


def _check_expert_skills_zero_duration(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if the candidate claims 'expert' or 'advanced' proficiency
    in 3 or more skills but with 0 months of usage duration.

    Example: A dataset trap has expert-level skills but 0 duration_months
    for all of them — clearly impossible.
    """
    skills = candidate.get("skills", [])

    zero_dur_expert_count = 0
    for skill in skills:
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0)

        if proficiency in ("expert", "advanced") and duration == 0:
            zero_dur_expert_count += 1

    if zero_dur_expert_count >= 3:
        return True, (
            f"Impossible skill profile: {zero_dur_expert_count} skills claimed as "
            f"expert/advanced with 0 months of duration"
        )

    return False, ""


def _check_experience_vs_career_mismatch(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if the stated years_of_experience is wildly different from the sum
    of career_history duration_months. A delta >5 years is suspicious.

    Example: CAND_0003430 claims 13.7 years but career history sums to 0.92 years.
    """
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience", 0)
    career = candidate.get("career_history", [])

    total_career_months = sum(job.get("duration_months", 0) for job in career)
    total_career_years = total_career_months / 12.0

    if abs(years_exp - total_career_years) > 5.0:
        return True, (
            f"Experience mismatch: stated {years_exp} years vs "
            f"career history sums to {total_career_years:.1f} years"
        )

    return False, ""
