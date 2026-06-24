# 🏆 Intelligent Candidate Discovery Engine
**Submission for the Redrob INDIA RUNS Hackathon**

This repository contains our complete, production-ready AI Recruiter engine designed to solve the **Intelligent Candidate Discovery & Ranking Challenge**. 

Our solution leverages a powerful hybrid architecture combining deep semantic understanding (via SentenceTransformers) with rigorous deterministic rule engines and behavioral signal multipliers. It successfully processes 100,000 raw candidate profiles, flags hidden "honeypots," and outputs a perfectly formatted CSV of the top 100 matches—all while strictly adhering to the 5-minute, CPU-only sandbox constraints.

---

## 🌟 The Architecture: 2-Stage Pipeline

To meet the rigorous compute constraints (CPU-only inference under 5 minutes) while maintaining high-quality semantic search, we designed a **Two-Stage Pipeline**:

1. **Stage 1: Offline Precomputation (`precompute.py`)**
   - **Environment:** Local development machine.
   - **Process:** Loads all 100,000 JSONL candidates. Converts their rich profiles (summaries, career histories, skills) into structured text and embeds them into 384-dimensional dense vectors using the HuggingFace `all-MiniLM-L6-v2` model.
   - **Output:** A highly optimized `candidate_embeddings.npy` file (~146 MB) and local model weights.

2. **Stage 2: Runtime CPU Ranking (`rank.py`)**
   - **Environment:** CPU-only sandbox, < 16GB RAM, No Network Access.
   - **Process:** Loads the local model weights and the `.npy` vector store directly into memory (taking < 1GB RAM). It embeds the target Job Description on the fly, computes heavily optimized cosine similarities using `numpy`, applies behavioral multipliers, and outputs the top 100 results.
   - **Performance:** Processes all 100,000 vectors and writes the submission CSV in **~66 seconds**.

---

## 🧠 The Secret Sauce: Intelligent Scoring

We didn't just stop at vector similarity. Our ranking algorithm (`src/ranker.py`) mimics a real expert human recruiter by blending 7 distinct signals into a final Composite Score:

1. **Semantic Similarity (35%)**: Cosine distance between the JD text and the candidate's career history/skills.
2. **Role Relevance (20%)**: Keyword matching isn't enough. We actively penalize "keyword-stuffed" profiles where the skills match but the actual job title is completely irrelevant (e.g., HR Managers with "Python" listed).
3. **Experience Alignment (15%)**: A mathematical Gaussian fit centered around the JD's exact sweet spot (5-9 years). Candidates with 4 or 10 years receive minor penalties; candidates with <3 or >12 years drop sharply.
4. **Behavioral Signals (15%)**: Aggregating Redrob activity signals—rewarding recent platform activity, high recruiter response rates, and sub-30-day notice periods.
5. **Core AI Skills (5%)**: Explicit bonus for containing JD-critical tools (e.g., Pinecone, FAISS, PyTorch, SentenceTransformers).
6. **Consulting Penalty (5%)**: Per the JD's specific instructions, candidates whose *entire* career has been at IT consulting/services firms receive a targeted penalty.
7. **Career Stability (5%)**: Penalizes "job-hoppers" averaging < 1.5-year tenures, rewarding 3+ year stability.

---

## 🕵️‍♂️ Catching the Honeypots

The dataset was laced with cleverly hidden trap profiles. Our engine successfully identified and **filtered out 65 honeypot candidates** during the ranking phase using three deterministic rules (`src/honeypot_detector.py`):

1. **The Time Traveler:** Job durations that wildly contradict the actual delta between `start_date` and `end_date`.
2. **The Fake Expert:** Candidates claiming "Expert" or "Advanced" proficiency in multiple hard skills while simultaneously showing `0 months` of usage duration.
3. **The Experience Paradox:** Profiles where the explicitly stated `years_of_experience` contradicts the total sum of their career history by more than 5 years.

---

## 💻 Local Setup & Reproduction

We have isolated our dependencies to ensure flawless local execution. 

### 1. Initialize Virtual Environment
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate the Vectors (Offline Precomputation)
*Note: This step is only required if you modify the base 100K dataset.*
```bash
python precompute.py
```

### 3. Generate the Hackathon Submission
This is the production CPU-only script.
```bash
python rank.py --candidates ../dataset/[PUB]\ India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl --out ./team_submission.csv
```

### 4. Validate the Output
We built an automated validation tool to ensure the CSV structure is 100% compliant with the judges' spec.
```bash
$env:PYTHONIOENCODING="utf-8"
python validate_submission.py
```

---

## 🎨 Bonus: Interactive Streamlit Dashboard

To showcase the power of the engine, we built a stunning, dark-mode web dashboard for recruiters. It allows you to paste *any* Job Description, tweak sliders for "Notice Period" or "Years of Experience", and instantly see the dynamic re-ranking of the 100,000 candidates alongside interactive Plotly analytics.

**Launch the dashboard:**
```bash
streamlit run app.py
```
*(Available locally at `http://localhost:8501`)*

---
*Built with ❤️ for the INDIA RUNS AI Hackathon.*
