"""
precompute.py — Offline script to generate Sentence-Transformer embeddings
for all 100K candidates and save the model weights locally.

This script runs offline on the development machine and is NOT
subject to the 5-minute CPU-only sandbox constraint.

Usage:
    python precompute.py
"""

import json
import os
import sys
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset",
    "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge")
CANDIDATES_FILE = os.path.join(DATASET_DIR, "candidates.jsonl")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "candidate_embeddings.npy")
IDS_FILE = os.path.join(DATA_DIR, "candidate_ids.json")
MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# Add src to path
sys.path.insert(0, SCRIPT_DIR)
from src.utils import build_candidate_text


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Step 1: Load and download model
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Save model locally for offline use
    logger.info(f"Saving model weights to {LOCAL_MODEL_PATH}")
    model.save(LOCAL_MODEL_PATH)
    logger.info("Model weights saved.")

    # Step 2: Load candidates
    logger.info(f"Loading candidates from {CANDIDATES_FILE}")
    candidates = []
    candidate_ids = []
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                candidates.append(c)
                candidate_ids.append(c["candidate_id"])

    logger.info(f"Loaded {len(candidates)} candidates.")

    # Step 3: Build text representations
    logger.info("Building text representations...")
    texts = []
    for i, c in enumerate(candidates):
        text = build_candidate_text(c)
        texts.append(text)
        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i + 1}/{len(candidates)} texts")

    # Step 4: Encode all texts
    logger.info("Encoding candidate texts (this may take a few minutes)...")
    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    elapsed = time.time() - start_time
    logger.info(f"Encoding complete in {elapsed:.1f}s. Shape: {embeddings.shape}")

    # Step 5: Save embeddings and IDs
    logger.info(f"Saving embeddings to {EMBEDDINGS_FILE} as float16")
    np.save(EMBEDDINGS_FILE, embeddings.astype(np.float16))

    logger.info(f"Saving candidate IDs to {IDS_FILE}")
    with open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)

    file_size_mb = os.path.getsize(EMBEDDINGS_FILE) / (1024 * 1024)
    logger.info(f"Done! Embeddings file size: {file_size_mb:.1f} MB")
    logger.info(f"Total candidates: {len(candidate_ids)}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
