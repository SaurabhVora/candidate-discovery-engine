"""
rank.py — CLI script to produce the submission CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

This script runs within the 5-minute CPU-only sandbox constraint by:
1. Loading precomputed candidate embeddings from data/candidate_embeddings.npy
2. Loading local model weights to embed only the JD text
3. Computing composite scores and producing the ranked top-100 CSV
"""

import argparse
import csv
import json
import os
import sys
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.utils import load_candidates_jsonl, build_candidate_text
from src.ranker import rank_candidates

# Paths
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "candidate_embeddings.npy")
IDS_FILE = os.path.join(DATA_DIR, "candidate_ids.json")
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")

# The Job Description text
JD_TEXT = """Senior AI Engineer — Founding Team at Redrob AI.
Own the intelligence layer: ranking, retrieval, and matching systems.
5-9 years experience, strong Python, production embeddings-based retrieval,
vector databases (Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch),
evaluation frameworks (NDCG, MRR, MAP, A/B testing).
Nice-to-have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost),
HR-tech/marketplace experience, distributed systems, open-source contributions.
Must be a builder who ships to production, not pure researcher.
Location: Pune/Noida preferred, open to Hyderabad/Mumbai/Delhi NCR.
Notice period: sub-30-day preferred. Culture: async-first, writes code,
moves fast. Ideal: 6-8 years, 4-5 in applied ML at product companies,
shipped ranking/search/recommendation system at scale.
Disqualifiers: pure research without production, only consulting firms
career, title-chasers switching every 1.5 years, purely CV/speech/robotics
without NLP/IR, only recent LangChain-based projects."""


def main():
    parser = argparse.ArgumentParser(description="Produce submission CSV")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    start_time = time.time()

    # Step 1: Load precomputed embeddings and candidate IDs
    logger.info("Loading precomputed embeddings...")
    candidate_embeddings = np.load(EMBEDDINGS_FILE)
    logger.info(f"Embeddings shape: {candidate_embeddings.shape}")

    with open(IDS_FILE, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)
    logger.info(f"Loaded {len(candidate_ids)} candidate IDs.")

    # Step 2: Load candidates
    candidates = load_candidates_jsonl(args.candidates)

    # Step 3: Embed the JD using local model
    logger.info(f"Loading local model from {LOCAL_MODEL_PATH}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(LOCAL_MODEL_PATH)
    jd_embedding = model.encode(JD_TEXT, convert_to_numpy=True, normalize_embeddings=False)
    logger.info(f"JD embedding shape: {jd_embedding.shape}")

    # Step 4: Rank
    results = rank_candidates(
        candidates=candidates,
        candidate_embeddings=candidate_embeddings,
        jd_embedding=jd_embedding,
        candidate_ids=candidate_ids,
        top_k=100,
    )

    # Step 5: Write CSV
    logger.info(f"Writing {len(results)} results to {args.out}")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                r["score"],
                r["reasoning"],
            ])

    elapsed = time.time() - start_time
    logger.info(f"Done! Total time: {elapsed:.1f}s")
    logger.info(f"Output: {args.out}")


if __name__ == "__main__":
    main()
