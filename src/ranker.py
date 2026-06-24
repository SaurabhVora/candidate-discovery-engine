"""
ranker.py — Core ranking engine that combines semantic similarity
with behavioral signal adjustments to produce the final candidate ranking.

Architecture:
1. Semantic Score: Cosine similarity between JD embedding and candidate embeddings
2. Experience Alignment: Gaussian penalty around the 5-9 year sweet spot
3. Behavioral Multiplier: Activity, responsiveness, notice period, location
4. Honeypot Filter: Exclude impossible profiles
5. Role Relevance: Penalize keyword-stuffed irrelevant titles
"""

import numpy as np
import logging
import math
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from src.honeypot_detector import detect_honeypot
from src.utils import (
    REFERENCE_DATE,
    build_candidate_text,
    count_core_ai_skills,
    has_only_consulting_history,
    is_title_relevant,
    is_title_irrelevant,
    is_in_target_location,
    compute_days_since_active,
    get_candidate_skill_names,
    CORE_AI_SKILLS,
)

logger = logging.getLogger(__name__)


def compute_experience_score(years_exp: float) -> float:
    """
    Gaussian-shaped score centered on 7 years (the JD's ideal).
    - 6-8 years: score ~1.0
    - 5-9 years: score ~0.85-0.95
    - 4 or 10 years: score ~0.7
    - <3 or >12 years: score drops sharply
    """
    ideal = 7.0
    sigma = 2.5
    score = math.exp(-0.5 * ((years_exp - ideal) / sigma) ** 2)
    return score


def compute_behavioral_score(candidate: Dict[str, Any]) -> float:
    """
    Compute a 0-1 behavioral score from Redrob signals.
    Higher score = more hireable/active candidate.
    """
    signals = candidate.get("redrob_signals", {})

    components = []
    weights = []

    # 1. Recruiter response rate (0-1) — heavily weighted
    rrr = signals.get("recruiter_response_rate", 0.0)
    components.append(rrr)
    weights.append(0.20)

    # 2. Activity recency — days since last active
    days_inactive = compute_days_since_active(candidate)
    if days_inactive <= 7:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.9
    elif days_inactive <= 90:
        recency = 0.7
    elif days_inactive <= 180:
        recency = 0.4
    else:
        recency = 0.1
    components.append(recency)
    weights.append(0.15)

    # 3. Open to work flag
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.4
    components.append(open_to_work)
    weights.append(0.10)

    # 4. Notice period
    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.8
    elif notice_days <= 90:
        notice_score = 0.6
    else:
        notice_score = 0.3
    components.append(notice_score)
    weights.append(0.10)

    # 5. Interview completion rate
    icr = signals.get("interview_completion_rate", 0.5)
    components.append(icr)
    weights.append(0.10)

    # 6. Response time (lower is better)
    avg_resp_time = signals.get("avg_response_time_hours", 48)
    if avg_resp_time <= 4:
        resp_time_score = 1.0
    elif avg_resp_time <= 12:
        resp_time_score = 0.85
    elif avg_resp_time <= 24:
        resp_time_score = 0.7
    elif avg_resp_time <= 48:
        resp_time_score = 0.5
    else:
        resp_time_score = 0.2
    components.append(resp_time_score)
    weights.append(0.10)

    # 7. Profile completeness
    pcs = signals.get("profile_completeness_score", 50) / 100.0
    components.append(pcs)
    weights.append(0.05)

    # 8. GitHub activity (normalized, -1 = no GitHub)
    github = signals.get("github_activity_score", -1)
    if github == -1:
        github_score = 0.3  # Neutral — no GitHub
    else:
        github_score = 0.3 + 0.7 * (github / 100.0)
    components.append(github_score)
    weights.append(0.05)

    # 9. Saved by recruiters
    saved = signals.get("saved_by_recruiters_30d", 0)
    saved_score = min(1.0, saved / 10.0)  # Cap at 10
    components.append(saved_score)
    weights.append(0.05)

    # 10. Verification signals
    verified = 0
    if signals.get("verified_email", False):
        verified += 1
    if signals.get("verified_phone", False):
        verified += 1
    if signals.get("linkedin_connected", False):
        verified += 1
    verify_score = verified / 3.0
    components.append(verify_score)
    weights.append(0.05)

    # 11. Offer acceptance rate
    oar = signals.get("offer_acceptance_rate", -1)
    if oar == -1:
        oar_score = 0.5  # Neutral
    else:
        oar_score = oar
    components.append(oar_score)
    weights.append(0.05)

    # Weighted sum
    total = sum(c * w for c, w in zip(components, weights))
    return total


def compute_role_relevance_score(candidate: Dict[str, Any]) -> float:
    """
    Assess how relevant the candidate's career history and title are
    to the Senior AI Engineer role. This catches keyword-stuffed profiles
    where skills match but the actual job role is completely unrelated.
    """
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    career = candidate.get("career_history", [])

    # Check current title
    if is_title_irrelevant(current_title):
        # Check if any career history has a relevant title
        has_relevant_history = False
        for job in career:
            if is_title_relevant(job.get("title", "")):
                has_relevant_history = True
                break
        if not has_relevant_history:
            return 0.1  # Heavily penalize — completely irrelevant career
        else:
            return 0.4  # Had relevant history but currently in irrelevant role

    if is_title_relevant(current_title):
        # Check career depth in relevant roles
        relevant_months = 0
        for job in career:
            if is_title_relevant(job.get("title", "")):
                relevant_months += job.get("duration_months", 0)
        relevant_years = relevant_months / 12.0

        if relevant_years >= 4:
            return 1.0
        elif relevant_years >= 2:
            return 0.85
        elif relevant_years >= 1:
            return 0.7
        else:
            return 0.6

    # Ambiguous title — give moderate score
    return 0.5


def compute_consulting_penalty(candidate: Dict[str, Any]) -> float:
    """
    Per JD: penalize candidates whose ENTIRE career is at consulting firms.
    If they have mixed experience (some product, some consulting), that's fine.
    """
    if has_only_consulting_history(candidate):
        return 0.3  # Heavy penalty
    return 1.0  # No penalty


def compute_career_stability_score(candidate: Dict[str, Any]) -> float:
    """
    Per JD: penalize title-chasers who switch companies every 1.5 years.
    Reward candidates who stay 3+ years.
    """
    career = candidate.get("career_history", [])
    if len(career) <= 1:
        return 0.7  # Too little data

    durations = [job.get("duration_months", 0) for job in career]
    avg_tenure = sum(durations) / len(durations)

    if avg_tenure >= 36:
        return 1.0  # Great stability
    elif avg_tenure >= 24:
        return 0.9
    elif avg_tenure >= 18:
        return 0.75
    elif avg_tenure >= 12:
        return 0.6
    else:
        return 0.4  # Job hopper


def rank_candidates(
    candidates: List[Dict[str, Any]],
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    candidate_ids: List[str],
    top_k: int = 100,
) -> List[Dict[str, Any]]:
    """
    Main ranking function. Produces a sorted list of top-K candidates
    with scores and reasoning.

    Args:
        candidates: List of full candidate dicts
        candidate_embeddings: (N, dim) numpy array of precomputed embeddings
        jd_embedding: (dim,) numpy array of the JD embedding
        candidate_ids: List of candidate_id strings matching embedding rows
        top_k: Number of top candidates to return

    Returns:
        List of dicts with: candidate_id, rank, score, reasoning, candidate
    """
    logger.info(f"Ranking {len(candidates)} candidates...")

    # Build a lookup map: candidate_id -> candidate dict
    cand_map = {c["candidate_id"]: c for c in candidates}

    # Step 1: Compute cosine similarities
    # Normalize embeddings for cosine similarity
    jd_norm = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-9)
    emb_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-9
    normed_embeddings = candidate_embeddings / emb_norms

    cosine_sims = normed_embeddings @ jd_norm  # (N,)

    # Step 2: Score each candidate
    scored = []
    honeypot_count = 0

    for i, cid in enumerate(candidate_ids):
        candidate = cand_map.get(cid)
        if candidate is None:
            continue

        # Honeypot check — skip if flagged
        is_honeypot, honeypot_reason = detect_honeypot(candidate)
        if is_honeypot:
            honeypot_count += 1
            continue

        semantic_score = float(cosine_sims[i])

        profile = candidate.get("profile", {})
        years_exp = profile.get("years_of_experience", 0)

        exp_score = compute_experience_score(years_exp)
        behavioral_score = compute_behavioral_score(candidate)
        role_score = compute_role_relevance_score(candidate)
        consulting_penalty = compute_consulting_penalty(candidate)
        stability_score = compute_career_stability_score(candidate)

        # Weighted composite score
        # Semantic similarity is the foundation, modified by other factors
        final_score = (
            0.35 * semantic_score +
            0.15 * exp_score +
            0.15 * behavioral_score +
            0.20 * role_score +
            0.05 * consulting_penalty +
            0.05 * stability_score +
            0.05 * min(1.0, count_core_ai_skills(candidate) / 5.0)
        )

        scored.append({
            "candidate_id": cid,
            "final_score": final_score,
            "semantic_score": semantic_score,
            "exp_score": exp_score,
            "behavioral_score": behavioral_score,
            "role_score": role_score,
            "consulting_penalty": consulting_penalty,
            "stability_score": stability_score,
            "candidate": candidate,
        })

    logger.info(f"Filtered out {honeypot_count} honeypot candidates.")

    # Step 3: Sort by final_score descending, then by candidate_id ascending for tiebreak
    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # Step 4: Take top K and assign ranks
    results = []
    for rank_idx, item in enumerate(scored[:top_k]):
        rank = rank_idx + 1
        candidate = item["candidate"]
        reasoning = generate_reasoning(candidate, item, rank)

        results.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": round(item["final_score"], 4),
            "reasoning": reasoning,
            "candidate": candidate,
            "semantic_score": item["semantic_score"],
            "behavioral_score": item["behavioral_score"],
            "exp_score": item["exp_score"],
            "role_score": item["role_score"],
        })

    return results


def generate_reasoning(candidate: Dict[str, Any], scores: Dict, rank: int) -> str:
    """
    Generate a specific, honest, non-templated reasoning string for
    why this candidate is at this rank. References real profile facts.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    name = profile.get("anonymized_name", "Unknown")
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    years = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")
    industry = profile.get("current_industry", "Unknown")

    notice = signals.get("notice_period_days", 0)
    rrr = signals.get("recruiter_response_rate", 0)
    open_to_work = signals.get("open_to_work_flag", False)
    github = signals.get("github_activity_score", -1)

    # Count relevant skills
    core_count = count_core_ai_skills(candidate)

    # Build reasoning parts
    parts = []

    # Opening — who they are
    parts.append(f"{title} at {company} ({industry}) with {years} years experience")

    # Location
    if is_in_target_location(candidate):
        parts.append(f"based in {location}, India (preferred location)")
    else:
        parts.append(f"located in {location}, {country}")

    # Strengths
    strengths = []
    if scores.get("semantic_score", 0) > 0.5:
        strengths.append("strong semantic match to JD requirements")
    if core_count >= 4:
        strengths.append(f"{core_count} core AI/ML skills")
    if scores.get("role_score", 0) >= 0.85:
        strengths.append("relevant career trajectory in tech/ML roles")
    if years >= 5 and years <= 9:
        strengths.append(f"{years} yrs falls in the 5-9 year sweet spot")

    if strengths:
        parts.append("Strengths: " + "; ".join(strengths))

    # Concerns
    concerns = []
    if notice > 60:
        concerns.append(f"{notice}-day notice period (JD prefers <30)")
    if rrr < 0.3:
        concerns.append(f"low recruiter response rate ({rrr:.0%})")
    if not open_to_work:
        concerns.append("not marked open to work")
    if years < 5:
        concerns.append(f"only {years} years experience (JD wants 5-9)")
    if years > 10:
        concerns.append(f"{years} years may exceed the 5-9 year range")
    if scores.get("consulting_penalty", 1.0) < 1.0:
        concerns.append("entire career at consulting firms")

    if concerns:
        parts.append("Concerns: " + "; ".join(concerns))

    return ". ".join(parts) + "."
