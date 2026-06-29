"""
app.py – Streamlit UI for CSV Insight Analyzer
------------------------------------------------
Run:  streamlit run app.py
      (main.py must be in the same folder)
"""

import os, io, sys, datetime
from pathlib import Path

# Ensure main.py is always findable regardless of working directory
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env from same folder as app.py
load_dotenv(dotenv_path=APP_DIR / ".env")

# Import core agent immediately so errors surface on startup not mid-run
try:
    from main import DataAnalysisAgent
    IMPORT_ERROR = None
except Exception as e:
    DataAnalysisAgent = None
    IMPORT_ERROR = str(e)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV Insight Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Show import error as a banner if main.py failed to load ──────────────────
if IMPORT_ERROR:
    st.error("Could not import main.py. Check it is in the same folder. Error: " + IMPORT_ERROR)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f1117;
    color: #e4e4e7;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #16181f;
    border-right: 1px solid #2a2d3a;
  }
  section[data-testid="stSidebar"] * { color: #c4c4cc !important; }

  /* ── Headers ── */
  h1 { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.6rem !important;
       letter-spacing: -0.5px; color: #a78bfa !important; }
  h2, h3 { color: #c4b5fd !important; }

  /* ── Cards ── */
  .card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .card-accent { border-left: 3px solid #7c3aed; }

  /* ── Metric chips ── */
  .chip {
    display: inline-block;
    background: #2a2d3a;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #a78bfa;
    margin: 2px;
  }

  /* ── Status badges ── */
  .badge-success { background:#14532d; color:#4ade80; border-radius:6px; padding:2px 10px; font-size:.8rem; }
  .badge-fail    { background:#450a0a; color:#f87171; border-radius:6px; padding:2px 10px; font-size:.8rem; }
  .badge-running { background:#1e3a5f; color:#60a5fa; border-radius:6px; padding:2px 10px; font-size:.8rem; }

  /* ── Input ── */
  textarea, input[type="text"] {
    background: #1a1d27 !important;
    border: 1px solid #3a3d4a !important;
    color: #e4e4e7 !important;
    border-radius: 8px !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: #7c3aed !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.4rem !important;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #6d28d9 !important; }

  /* ── Dataframe ── */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* ── Code block ── */
  .stCodeBlock { border-radius: 8px !important; }

  /* ── Divider ── */
  hr { border-color: #2a2d3a !important; }

  /* ── Scrollable history ── */
  .history-box {
    max-height: 320px;
    overflow-y: auto;
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
  }
  .history-item {
    padding: 0.5rem 0;
    border-bottom: 1px solid #2a2d3a;
    font-size: 0.85rem;
  }
  .history-item:last-child { border-bottom: none; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
for key, val in {
    "all_dfs":   {},
    "metadata":  {},
    "history":   [],       # list of result dicts
    "agent":     None,
    "api_ok":    False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Lazy agent import (after .env is loaded) ──────────────────────────────────
def get_agent():
    if IMPORT_ERROR:
        st.error(f"Cannot load main.py: {IMPORT_ERROR}")
        return None
    if not st.session_state.all_dfs:
        return None
    return DataAnalysisAgent(st.session_state.all_dfs, st.session_state.metadata)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 CSV Insight")
    st.markdown("---")

    # API key check (silent – no badge shown)
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        st.session_state.api_ok = True
    else:
        manual_key = st.text_input("Paste Gemini API key", type="password")
        if manual_key:
            os.environ["GEMINI_API_KEY"] = manual_key
            st.session_state.api_ok = True

    st.markdown("---")
    st.markdown("### 📂 Upload CSVs")
    st.caption("Up to 3 files")

    uploaded = st.file_uploader(
        "Choose CSV files",
        type=["csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        new_dfs, new_meta = {}, {}
        for i, f in enumerate(uploaded[:3]):
            key = "data" if i == 0 else f"data{i+1}"
            df  = pd.read_csv(f)
            new_dfs[key]  = df
            new_meta[key] = {c: str(t) for c, t in df.dtypes.items()}

        if set(new_dfs.keys()) != set(st.session_state.all_dfs.keys()):
            st.session_state.all_dfs  = new_dfs
            st.session_state.metadata = new_meta
            st.session_state.history  = []
            st.session_state.agent    = None
            st.success(f"{len(new_dfs)} file(s) loaded.")

    st.markdown("---")
    st.markdown("### 📊 Loaded DataFrames")
    if st.session_state.all_dfs:
        for name, df in st.session_state.all_dfs.items():
            st.markdown(
                f'<span class="chip">{name}</span>'
                f'<span class="chip">{len(df):,} rows</span>'
                f'<span class="chip">{len(df.columns)} cols</span>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No files loaded yet.")

    st.markdown("---")
    if st.button("🗑 Clear history"):
        st.session_state.history = []
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# CSV Insight Analyzer")
st.markdown("*Natural-language data analysis powered by Gemini + smolagents*")
st.markdown("---")

# ── Data preview ─────────────────────────────────────────────────────────────
if st.session_state.all_dfs:
    with st.expander("🗃 Data Preview", expanded=False):
        tabs = st.tabs(list(st.session_state.all_dfs.keys()))
        for tab, (name, df) in zip(tabs, st.session_state.all_dfs.items()):
            with tab:
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{len(df):,}")
                col2.metric("Columns", len(df.columns))
                col3.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                st.dataframe(df.head(10), use_container_width=True)

                # Schema table
                schema_df = pd.DataFrame({
                    "Column":   df.columns,
                    "Type":     df.dtypes.astype(str).values,
                    "Non-null": df.notna().sum().values,
                    "Unique":   df.nunique().values,
                })
                st.markdown("**Schema**")
                st.dataframe(schema_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Analysis input ────────────────────────────────────────────────────────
    st.markdown("### 💬 Ask a question about your data")

    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        goal = st.text_area(
            "goal",
            placeholder="e.g. Show the top 5 products by revenue  |  Find missing values in each column  |  Correlation between Age and Salary",
            height=90,
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶ Run", use_container_width=True)

    # ── Suggested prompts ─────────────────────────────────────────────────────
    st.markdown("**Quick starters:**")
    qcols = st.columns(4)
    starters = [
        "Show missing values per column",
        "Top 10 rows by highest numeric value",
        "Value counts for all categorical columns",
        "Descriptive statistics summary",
    ]
    for qc, starter in zip(qcols, starters):
        if qc.button(starter, key=f"q_{starter}"):
            goal = starter
            run_btn = True

    # ── Run analysis ──────────────────────────────────────────────────────────
    if run_btn and goal.strip():
        if not st.session_state.api_ok:
            st.error("Add your GEMINI_API_KEY to .env or paste it in the sidebar.")
        else:
            with st.spinner("🧠 Analyzing…"):
                agent  = get_agent()
                result = agent.run(goal.strip())
                st.session_state.history.insert(0, result)
                st.rerun()

    st.markdown("---")

    # ── Results history ───────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("### 📋 Results")
        for i, res in enumerate(st.session_state.history):
            is_ok = res["status"] == "Success"
            badge = '<span class="badge-success">✓ Success</span>' if is_ok else '<span class="badge-fail">✗ Failed</span>'

            with st.expander(
                f"{'✅' if is_ok else '❌'}  {res['goal'][:80]}",
                expanded=(i == 0),
            ):
                st.markdown(badge, unsafe_allow_html=True)
                st.caption(f"🕐 {res['timestamp']}")

                if is_ok:
                    # ── Interpretation ────────────────────────────────────────
                    st.markdown(
                        f'<div class="card card-accent">'
                        f'<b>📊 Insight</b><br><br>{res["interpretation"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Output file ───────────────────────────────────────────
                    if res.get("output_file") and Path(res["output_file"]).exists():
                        out_path = Path(res["output_file"])
                        if out_path.suffix == ".csv":
                            out_df = pd.read_csv(out_path)
                            st.markdown("**Output table** (`result.csv`)")
                            st.dataframe(out_df, use_container_width=True)

                            st.download_button(
                                "⬇ Download result.csv",
                                data=out_df.to_csv(index=False).encode(),
                                file_name=f"result_{res['timestamp']}.csv",
                                mime="text/csv",
                                key=f"dl_{i}",
                            )
                        elif out_path.suffix in (".png", ".jpg", ".pdf"):
                            st.image(str(out_path), caption="Generated plot")

                    # ── Generated code ────────────────────────────────────────
                    if res.get("code_path") and Path(res["code_path"]).exists():
                        with st.expander("🧾 View generated code"):
                            st.code(
                                Path(res["code_path"]).read_text(encoding="utf-8"),
                                language="python",
                            )

                else:
                    # ── Error detail ──────────────────────────────────────────
                    st.markdown(
                        f'<div class="card">'
                        f'<b>Error</b><br><pre style="color:#f87171;font-size:.8rem;white-space:pre-wrap;">'
                        f'{res["error"][:800]}</pre></div>',
                        unsafe_allow_html=True,
                    )

        # ── Export full history ───────────────────────────────────────────────
        if len(st.session_state.history) > 1:
            hist_df = pd.DataFrame([
                {"goal": r["goal"], "status": r["status"], "timestamp": r["timestamp"]}
                for r in st.session_state.history
            ])
            st.download_button(
                "⬇ Export session history",
                data=hist_df.to_csv(index=False).encode(),
                file_name="session_history.csv",
                mime="text/csv",
            )

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 4rem 0; color: #52525b;">
      <div style="font-size:3rem;">🔬</div>
      <h3 style="color:#52525b; font-family:'IBM Plex Mono',monospace;">Upload a CSV to get started</h3>
      <p>Use the sidebar to load up to 3 CSV files, then ask anything about your data.</p>
    </div>
    """, unsafe_allow_html=True)