"""
utils.py — File loading, text assembly, and helper utilities
for the Intelligent Candidate Discovery Engine.
"""

import json
import os
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Reference date for computing recency signals
REFERENCE_DATE = datetime(2026, 6, 24)

# Known consulting/services companies (per JD disqualifiers)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl technologies", "hcl",
    "tech mahindra",
    "deloitte",
    "ey", "ernst & young",
    "kpmg",
    "pwc", "pricewaterhousecoopers",
}

# AI/ML relevant titles (loose matching)
AI_ML_RELEVANT_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "data scientist", "research scientist", "applied scientist",
    "nlp engineer", "deep learning engineer", "software engineer",
    "backend engineer", "senior engineer", "senior software engineer",
    "principal engineer", "staff engineer", "tech lead",
    "data engineer", "analytics engineer", "search engineer",
    "platform engineer", "infrastructure engineer",
    "junior ml engineer", "senior machine learning engineer",
    "senior ai engineer", "senior data scientist",
    "lead data scientist", "lead ml engineer",
    "full stack engineer", "fullstack engineer",
    ".net developer", "python developer", "software developer",
    "devops engineer", "sre", "site reliability engineer",
    "frontend engineer",
}

# Completely irrelevant titles for an AI Engineering role
IRRELEVANT_TITLES = {
    "hr manager", "human resources", "recruiter",
    "marketing manager", "content writer", "content strategist",
    "graphic designer", "ui designer", "ux designer",
    "accountant", "finance manager", "accounts manager",
    "sales executive", "sales manager", "business development",
    "operations manager", "operations executive",
    "customer support", "customer service",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "teacher", "professor", "lecturer",
    "legal counsel", "lawyer", "advocate",
    "medical", "doctor", "nurse",
}

# Core AI/ML skills that the JD explicitly values
CORE_AI_SKILLS = {
    "embeddings", "sentence-transformers", "sentence transformers",
    "vector database", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "elasticsearch", "opensearch",
    "ranking", "retrieval", "information retrieval", "search",
    "recommendation", "recommendation system",
    "nlp", "natural language processing", "text mining",
    "machine learning", "ml", "deep learning", "dl",
    "python", "pytorch", "tensorflow",
    "transformers", "huggingface", "hugging face",
    "bert", "gpt", "llm", "large language model",
    "rag", "retrieval augmented generation",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "xgboost", "learning to rank", "l2r",
    "ndcg", "mrr", "map", "a/b testing", "ab testing",
    "data pipeline", "airflow", "spark", "kafka",
    "sql", "nosql", "mongodb", "redis",
    "docker", "kubernetes", "aws", "gcp", "azure",
    "scikit-learn", "sklearn", "scipy", "numpy", "pandas",
    "mlflow", "wandb", "weights & biases", "weights and biases",
    "openai", "anthropic", "langchain",
    "bge", "e5", "bi-encoder", "cross-encoder",
    "distributed systems", "microservices",
}


def load_candidates_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load all candidates from a JSONL file."""
    candidates = []
    logger.info(f"Loading candidates from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    logger.info(f"Loaded {len(candidates)} candidates.")
    return candidates


def load_candidates_json(filepath: str) -> List[Dict[str, Any]]:
    """Load all candidates from a JSON array file."""
    logger.info(f"Loading candidates from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    logger.info(f"Loaded {len(candidates)} candidates.")
    return candidates


def build_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Assemble a rich text representation from a candidate's profile,
    career history, education, and skills. This is what gets embedded.
    """
    parts = []

    # Profile section
    profile = candidate.get("profile", {})
    if profile.get("headline"):
        parts.append(f"Headline: {profile['headline']}")
    if profile.get("summary"):
        parts.append(f"Summary: {profile['summary']}")
    if profile.get("current_title"):
        parts.append(f"Current Title: {profile['current_title']}")
    if profile.get("current_company"):
        parts.append(f"Current Company: {profile['current_company']}")
    if profile.get("current_industry"):
        parts.append(f"Industry: {profile['current_industry']}")
    if profile.get("years_of_experience") is not None:
        parts.append(f"Years of Experience: {profile['years_of_experience']}")

    # Career history
    career = candidate.get("career_history", [])
    for i, job in enumerate(career):
        job_parts = []
        if job.get("title"):
            job_parts.append(job["title"])
        if job.get("company"):
            job_parts.append(f"at {job['company']}")
        if job.get("industry"):
            job_parts.append(f"({job['industry']})")
        if job.get("duration_months"):
            job_parts.append(f"for {job['duration_months']} months")
        if job.get("description"):
            job_parts.append(f"- {job['description']}")
        if job_parts:
            parts.append(f"Career {i+1}: {' '.join(job_parts)}")

    # Education
    education = candidate.get("education", [])
    for edu in education:
        edu_parts = []
        if edu.get("degree"):
            edu_parts.append(edu["degree"])
        if edu.get("field_of_study"):
            edu_parts.append(f"in {edu['field_of_study']}")
        if edu.get("institution"):
            edu_parts.append(f"from {edu['institution']}")
        if edu_parts:
            parts.append(f"Education: {' '.join(edu_parts)}")

    # Skills
    skills = candidate.get("skills", [])
    skill_names = [s["name"] for s in skills if s.get("name")]
    if skill_names:
        parts.append(f"Skills: {', '.join(skill_names)}")

    # Certifications
    certs = candidate.get("certifications", [])
    cert_names = [c["name"] for c in certs if c.get("name")]
    if cert_names:
        parts.append(f"Certifications: {', '.join(cert_names)}")

    return "\n".join(parts)


def get_candidate_skill_names(candidate: Dict[str, Any]) -> set:
    """Extract a set of lowercase skill names from a candidate."""
    skills = candidate.get("skills", [])
    return {s["name"].lower().strip() for s in skills if s.get("name")}


def is_title_relevant(title: str) -> bool:
    """Check if a job title is relevant to AI/ML engineering."""
    title_lower = title.lower().strip()
    for relevant in AI_ML_RELEVANT_TITLES:
        if relevant in title_lower:
            return True
    return False


def is_title_irrelevant(title: str) -> bool:
    """Check if a job title is clearly irrelevant to AI engineering."""
    title_lower = title.lower().strip()
    for irrelevant in IRRELEVANT_TITLES:
        if irrelevant in title_lower:
            return True
    return False


def has_only_consulting_history(candidate: Dict[str, Any]) -> bool:
    """
    Check if a candidate has ONLY worked at consulting/services firms.
    Per JD: candidates with prior product-company experience are fine.
    """
    career = candidate.get("career_history", [])
    if not career:
        return False

    for job in career:
        company = job.get("company", "").lower().strip()
        is_consulting = False
        for firm in CONSULTING_FIRMS:
            if firm in company:
                is_consulting = True
                break
        if not is_consulting:
            return False  # Found at least one non-consulting company

    return True  # All companies are consulting firms


def compute_days_since_active(candidate: Dict[str, Any]) -> int:
    """Compute number of days since last_active_date relative to REFERENCE_DATE."""
    signals = candidate.get("redrob_signals", {})
    last_active = signals.get("last_active_date")
    if not last_active:
        return 999  # Very stale
    try:
        active_dt = datetime.strptime(last_active, "%Y-%m-%d")
        delta = (REFERENCE_DATE - active_dt).days
        return max(0, delta)
    except (ValueError, TypeError):
        return 999


def count_core_ai_skills(candidate: Dict[str, Any]) -> int:
    """Count how many core AI/ML skills the candidate has."""
    candidate_skills = get_candidate_skill_names(candidate)
    count = 0
    for skill in candidate_skills:
        for core in CORE_AI_SKILLS:
            if core in skill or skill in core:
                count += 1
                break
    return count


def is_in_target_location(candidate: Dict[str, Any]) -> bool:
    """Check if candidate is in or near one of the JD's preferred locations."""
    profile = candidate.get("profile", {})
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()

    target_cities = ["pune", "noida", "hyderabad", "mumbai", "delhi",
                     "ncr", "bangalore", "bengaluru", "gurgaon", "gurugram"]

    if country != "india":
        return False

    for city in target_cities:
        if city in location:
            return True
    return False
