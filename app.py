"""
app.py — Streamlit Frontend Dashboard for the Intelligent Candidate Discovery Engine.

A recruiter can paste a Job Description and instantly see a ranked shortlist
of the best-fit candidates with clear matching scores and reasoning.

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import sys
import time
import plotly.graph_objects as go
import plotly.express as px

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.utils import (
    load_candidates_jsonl,
    load_candidates_json,
    build_candidate_text,
    count_core_ai_skills,
    is_in_target_location,
    compute_days_since_active,
    REFERENCE_DATE,
)
from src.ranker import rank_candidates
from src.honeypot_detector import detect_honeypot

# Paths
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset",
    "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "candidate_embeddings.npy")
IDS_FILE = os.path.join(DATA_DIR, "candidate_ids.json")
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")
CANDIDATES_FILE = os.path.join(DATASET_DIR, "candidates.jsonl")
SAMPLE_FILE = os.path.join(DATASET_DIR, "sample_candidates.json")

# Default JD
DEFAULT_JD = """Senior AI Engineer — Founding Team at Redrob AI.
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
shipped ranking/search/recommendation system at scale."""

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Redrob AI — Candidate Discovery Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .hero-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 30, 60, 0.9), rgba(20, 20, 50, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* Candidate cards */
    .candidate-card {
        background: linear-gradient(135deg, rgba(25, 25, 55, 0.95), rgba(15, 15, 40, 0.98));
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }

    .candidate-card:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 6px 28px rgba(102, 126, 234, 0.15);
        transform: translateY(-1px);
    }

    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.1rem;
        color: white;
        margin-right: 1rem;
        flex-shrink: 0;
    }

    .rank-top5 { background: linear-gradient(135deg, #f093fb, #f5576c); }
    .rank-top10 { background: linear-gradient(135deg, #667eea, #764ba2); }
    .rank-top25 { background: linear-gradient(135deg, #4facfe, #00f2fe); color: #0a0a2e; }
    .rank-rest { background: rgba(255, 255, 255, 0.1); }

    .candidate-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
    }

    .candidate-headline {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        margin-top: 0.15rem;
    }

    .score-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
    }

    .score-high { background: rgba(76, 175, 80, 0.2); color: #81c784; border: 1px solid rgba(76, 175, 80, 0.3); }
    .score-med { background: rgba(255, 183, 77, 0.2); color: #ffb74d; border: 1px solid rgba(255, 183, 77, 0.3); }
    .score-low { background: rgba(229, 115, 115, 0.2); color: #e57373; border: 1px solid rgba(229, 115, 115, 0.3); }

    .reasoning-text {
        color: rgba(255, 255, 255, 0.65);
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    .signal-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
        gap: 0.5rem;
        margin-top: 0.6rem;
    }

    .signal-item {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        padding: 0.5rem 0.7rem;
        font-size: 0.78rem;
    }

    .signal-label { color: rgba(255, 255, 255, 0.45); font-weight: 500; }
    .signal-value { color: rgba(255, 255, 255, 0.85); font-weight: 600; margin-top: 0.15rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ==================== CACHING ====================
@st.cache_resource
def load_model():
    """Load the SentenceTransformer model from local weights."""
    from sentence_transformers import SentenceTransformer
    if os.path.exists(LOCAL_MODEL_PATH):
        return SentenceTransformer(LOCAL_MODEL_PATH)
    else:
        st.warning("⚠️ Local model not found. Downloading from HuggingFace...")
        return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_embeddings():
    """Load precomputed embeddings and candidate IDs."""
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    with open(IDS_FILE, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return embeddings, ids


@st.cache_data
def load_all_candidates():
    """Load all candidates from JSONL."""
    if os.path.exists(CANDIDATES_FILE):
        return load_candidates_jsonl(CANDIDATES_FILE)
    elif os.path.exists(SAMPLE_FILE):
        return load_candidates_json(SAMPLE_FILE)
    else:
        st.error("No candidate data found!")
        return []


def get_rank_class(rank):
    if rank <= 5:
        return "rank-top5"
    elif rank <= 10:
        return "rank-top10"
    elif rank <= 25:
        return "rank-top25"
    return "rank-rest"


def get_score_class(score):
    if score >= 0.7:
        return "score-high"
    elif score >= 0.5:
        return "score-med"
    return "score-low"


# ==================== MAIN APP ====================
def main():
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>🔍 Intelligent Candidate Discovery Engine</h1>
        <p>Paste a Job Description below and discover the best-fit candidates with AI-powered semantic matching and behavioral analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.markdown("---")
        st.markdown("### 📋 Job Description")
        jd_text = st.text_area(
            "Paste your JD here:",
            value=DEFAULT_JD,
            height=300,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🎛️ Filters")

        max_notice = st.slider("Max Notice Period (days)", 0, 180, 90)
        min_exp = st.slider("Min Experience (years)", 0.0, 20.0, 0.0, 0.5)
        max_exp = st.slider("Max Experience (years)", 0.0, 30.0, 20.0, 0.5)
        only_open_to_work = st.checkbox("Only Open to Work", value=False)
        only_india = st.checkbox("Only India-based", value=False)

        st.markdown("---")
        run_search = st.button("🚀 Run Candidate Search", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("### 📊 Export")
        export_csv = st.button("📥 Download Results as CSV", use_container_width=True)

    # Check if data is available
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(IDS_FILE):
        st.warning(
            "⚠️ Precomputed embeddings not found. "
            "Please run `python precompute.py` first to generate embeddings."
        )
        st.stop()

    # Load data
    model = load_model()
    embeddings, candidate_ids = load_embeddings()
    candidates = load_all_candidates()

    # Run search
    if run_search or "results" not in st.session_state:
        with st.spinner("🧠 Embedding JD and computing semantic matches..."):
            start_time = time.time()

            jd_embedding = model.encode(jd_text, convert_to_numpy=True, normalize_embeddings=False)

            results = rank_candidates(
                candidates=candidates,
                candidate_embeddings=embeddings,
                jd_embedding=jd_embedding,
                candidate_ids=candidate_ids,
                top_k=200,  # Get extra for filtering
            )

            elapsed = time.time() - start_time

        # Apply filters
        filtered = []
        for r in results:
            c = r["candidate"]
            signals = c.get("redrob_signals", {})
            profile = c.get("profile", {})

            if signals.get("notice_period_days", 0) > max_notice:
                continue
            if profile.get("years_of_experience", 0) < min_exp:
                continue
            if profile.get("years_of_experience", 0) > max_exp:
                continue
            if only_open_to_work and not signals.get("open_to_work_flag", False):
                continue
            if only_india and profile.get("country", "").lower() != "india":
                continue

            filtered.append(r)

        # Re-rank after filtering
        for i, r in enumerate(filtered[:100]):
            r["rank"] = i + 1

        st.session_state["results"] = filtered[:100]
        st.session_state["elapsed"] = elapsed
        st.session_state["total_candidates"] = len(candidates)

    results = st.session_state.get("results", [])
    elapsed = st.session_state.get("elapsed", 0)
    total_candidates = st.session_state.get("total_candidates", 0)

    if not results:
        st.info("No candidates match the current filters. Try adjusting your criteria.")
        st.stop()

    # ==================== METRICS ====================
    col1, col2, col3, col4, col5 = st.columns(5)

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    top10_avg = sum(r["score"] for r in results[:10]) / min(10, len(results)) if results else 0
    open_count = sum(1 for r in results if r["candidate"].get("redrob_signals", {}).get("open_to_work_flag", False))
    india_count = sum(1 for r in results if r["candidate"].get("profile", {}).get("country", "").lower() == "india")

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_candidates:,}</div>
            <div class="metric-label">Pool Size</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(results)}</div>
            <div class="metric-label">Shortlisted</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top10_avg:.3f}</div>
            <div class="metric-label">Top-10 Avg Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{open_count}</div>
            <div class="metric-label">Open to Work</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{elapsed:.1f}s</div>
            <div class="metric-label">Search Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==================== CHARTS ====================
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Score distribution
        scores = [r["score"] for r in results]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=scores,
            nbinsx=20,
            marker_color="rgba(102, 126, 234, 0.7)",
            marker_line_color="rgba(102, 126, 234, 1)",
            marker_line_width=1,
        ))
        fig_dist.update_layout(
            title="Score Distribution",
            xaxis_title="Composite Score",
            yaxis_title="Candidates",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.5)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with chart_col2:
        # Experience vs Score scatter
        exp_data = [(r["candidate"]["profile"].get("years_of_experience", 0), r["score"]) for r in results]
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=[d[0] for d in exp_data],
            y=[d[1] for d in exp_data],
            mode="markers",
            marker=dict(
                size=8,
                color=[d[1] for d in exp_data],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Score"),
            ),
        ))
        fig_scatter.update_layout(
            title="Experience vs Match Score",
            xaxis_title="Years of Experience",
            yaxis_title="Composite Score",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.5)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ==================== CANDIDATE CARDS ====================
    st.markdown("### 🏆 Ranked Candidate Shortlist")

    for r in results:
        c = r["candidate"]
        profile = c.get("profile", {})
        signals = c.get("redrob_signals", {})
        rank = r["rank"]
        score = r["score"]

        rank_class = get_rank_class(rank)
        score_class = get_score_class(score)

        # Build signal items
        notice = signals.get("notice_period_days", 0)
        rrr = signals.get("recruiter_response_rate", 0)
        github = signals.get("github_activity_score", -1)
        open_flag = "✅ Yes" if signals.get("open_to_work_flag", False) else "❌ No"
        days_active = compute_days_since_active(c)
        core_skills = count_core_ai_skills(c)

        github_display = f"{github}/100" if github >= 0 else "N/A"
        active_display = f"{days_active}d ago" if days_active < 999 else "Unknown"

        st.markdown(f"""
        <div class="candidate-card">
            <div style="display: flex; align-items: flex-start;">
                <div class="rank-badge {rank_class}">#{rank}</div>
                <div style="flex: 1;">
                    <div class="candidate-name">{profile.get('anonymized_name', 'N/A')}</div>
                    <div class="candidate-headline">{profile.get('headline', '')}</div>
                    <div style="margin-top: 0.6rem;">
                        <span class="score-pill {score_class}">Score: {score:.4f}</span>
                        <span class="score-pill" style="background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.7); border: 1px solid rgba(255,255,255,0.1);">
                            {profile.get('years_of_experience', 0)} yrs
                        </span>
                        <span class="score-pill" style="background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.7); border: 1px solid rgba(255,255,255,0.1);">
                            📍 {profile.get('location', 'N/A')}, {profile.get('country', '')}
                        </span>
                        <span class="score-pill" style="background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.7); border: 1px solid rgba(255,255,255,0.1);">
                            🏢 {profile.get('current_company', 'N/A')}
                        </span>
                    </div>
                    <div class="signal-grid">
                        <div class="signal-item">
                            <div class="signal-label">Notice</div>
                            <div class="signal-value">{notice} days</div>
                        </div>
                        <div class="signal-item">
                            <div class="signal-label">Response Rate</div>
                            <div class="signal-value">{rrr:.0%}</div>
                        </div>
                        <div class="signal-item">
                            <div class="signal-label">GitHub</div>
                            <div class="signal-value">{github_display}</div>
                        </div>
                        <div class="signal-item">
                            <div class="signal-label">Open to Work</div>
                            <div class="signal-value">{open_flag}</div>
                        </div>
                        <div class="signal-item">
                            <div class="signal-label">Last Active</div>
                            <div class="signal-value">{active_display}</div>
                        </div>
                        <div class="signal-item">
                            <div class="signal-label">Core AI Skills</div>
                            <div class="signal-value">{core_skills}</div>
                        </div>
                    </div>
                    <div class="reasoning-text">
                        💡 {r['reasoning']}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==================== EXPORT ====================
    if export_csv:
        csv_data = []
        for r in results:
            csv_data.append({
                "candidate_id": r["candidate_id"],
                "rank": r["rank"],
                "score": r["score"],
                "reasoning": r["reasoning"],
            })
        df = pd.DataFrame(csv_data)
        csv_string = df.to_csv(index=False)

        st.download_button(
            label="📥 Download CSV",
            data=csv_string,
            file_name="submission.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
