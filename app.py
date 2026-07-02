"""
app.py – Streamlit UI for Insyra
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
    page_title="Insyra",
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

  /* ── Buttons (primary = violet, e.g. Run / Quick starters) ── */
  .stButton > button[kind="primary"],
  .stButton > button[data-testid="baseButton-primary"] {
    background: #7c3aed !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.4rem !important;
    transition: background 0.2s;
  }
  .stButton > button[kind="primary"]:hover,
  .stButton > button[data-testid="baseButton-primary"]:hover { background: #6d28d9 !important; }

  /* ── Buttons (secondary = neutral, e.g. the ✕ remove-file button) ── */
  .stButton > button[kind="secondary"],
  .stButton > button[data-testid="baseButton-secondary"] {
    background: #2a2d3a !important;
    color: #c4c4cc !important;
    border: 1px solid #3a3d4a !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: background 0.2s;
  }
  .stButton > button[kind="secondary"]:hover,
  .stButton > button[data-testid="baseButton-secondary"]:hover {
    background: #3a3d4a !important;
    color: #f87171 !important;
    border-color: #f87171 !important;
  }

  /* ── Buttons (default/no type set, e.g. Quick starters, Clear history) ── */
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

  /* ── Hide native file-uploader preview cards ──
       We render our own styled file chips in "Loaded DataFrames" instead,
       so the default Streamlit thumbnail/remove card is redundant. */
  [data-testid="stFileUploaderFile"] {
    display: none !important;
  }
  [data-testid="stFileUploaderFileName"] {
    display: none !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
for key, val in {
    "all_dfs":            {},
    "metadata":           {},
    "file_names":         {},      
    "history":            [],       
    "agent":              None,
    "api_ok":             False,
    "goal_input_value":   "",       
    "clear_goal_pending": False,    
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.clear_goal_pending:
    st.session_state.goal_input_value = ""
    st.session_state.clear_goal_pending = False


# ── Lazy agent import (after .env is loaded) ──────────────────────────────────
def get_agent():
    if IMPORT_ERROR:
        st.error(f"Cannot load main.py: {IMPORT_ERROR}")
        return None
    if not st.session_state.all_dfs:
        return None
    # Inject original filenames into metadata so describe questions show real names
    meta = dict(st.session_state.metadata)
    meta["__filenames__"] = st.session_state.file_names
    return DataAnalysisAgent(
        st.session_state.all_dfs,
        meta,
        file_names=st.session_state.get("file_names", {}),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Insyra")
    st.markdown("---")

    # API key check (silent – no badge or divider shown when key is already set)
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

    # ── Track up to 3 individual file slots ───────────────────────────────────
    if "csv_slots" not in st.session_state:
        st.session_state.csv_slots = [None, None, None]  
        st.session_state.uploader_gen = 0   

    n_filled = sum(1 for s in st.session_state.csv_slots if s is not None)

    if n_filled < 3:
        label = "Upload" if n_filled == 0 else "Upload more"
        # Key includes both the slot index AND a generation counter.
        # Bumping uploader_gen after each successful upload forces Streamlit
        # to mount a brand-new widget instance instead of reusing one that
        # still holds the previously-selected file internally.
        uploader_key = f"slot_uploader_{n_filled}_{st.session_state.uploader_gen}"
        new_file = st.file_uploader(
            label,
            type=["csv"],
            accept_multiple_files=False,
            key=uploader_key,
        )
        if new_file is not None:
            st.session_state.csv_slots[n_filled] = new_file
            st.session_state.uploader_gen += 1   
            st.rerun()

    # ── Rebuild all_dfs from filled slots whenever slots change ───────────────
    filled_files = [f for f in st.session_state.csv_slots if f is not None]
    if filled_files:
        new_dfs, new_meta, new_names = {}, {}, {}
        read_error = None
        for i, f in enumerate(filled_files):
            key = "data" if i == 0 else f"data{i+1}"
            f.seek(0)   
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                read_error = f"'{f.name}' appears to be empty — please upload a CSV with data."
                break
            new_dfs[key]   = df
            new_meta[key]  = {c: str(t) for c, t in df.dtypes.items()}
            new_names[key] = f.name

        if read_error:
            st.error(read_error)
        else:
            st.session_state.file_names = new_names
            if set(new_dfs.keys()) != set(st.session_state.all_dfs.keys()):
                st.session_state.all_dfs  = new_dfs
                st.session_state.metadata = new_meta
                st.session_state.history  = []
                st.session_state.agent    = None

    st.markdown("---")
    st.markdown("### 📊 Loaded DataFrames")
    if filled_files:
        for i, f in enumerate(filled_files):
            name = "data" if i == 0 else f"data{i+1}"
            df   = st.session_state.all_dfs.get(name)
            if df is None:
                continue

            row_col, btn_col = st.columns([5, 1])
            with row_col:
                st.markdown(f'📄 **{f.name}**', unsafe_allow_html=False)
                st.markdown(
                    f'<span class="chip">{len(df):,} rows</span>'
                    f'<span class="chip">{len(df.columns)} cols</span>',
                    unsafe_allow_html=True,
                )
            with btn_col:
                if st.button("✕", key=f"remove_slot_{i}", type="primary"):
                    st.session_state.csv_slots[i] = None
                    remaining = [s for s in st.session_state.csv_slots if s is not None]
                    st.session_state.csv_slots = remaining + [None] * (3 - len(remaining))
                    st.session_state.uploader_gen += 1   
                    st.rerun()
    else:
        st.caption("No files loaded yet.")


# ── Main layout ───────────────────────────────────────────────────────────────
title_col, clear_col = st.columns([8, 2])
with title_col:
    st.markdown("# Insyra")
with clear_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.history:
        if st.button("🗑 Clear history", key="clear_history_top", use_container_width=True):
            st.session_state.history = []
            st.rerun()
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
            key="goal_input_value",
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
    starter_clicked = None
    for qc, starter in zip(qcols, starters):
        if qc.button(starter, key=f"q_{starter}"):
            starter_clicked = starter

    # ── Run analysis ──────────────────────────────────────────────────────────
    run_goal = starter_clicked if starter_clicked else (goal.strip() if run_btn else None)

    if run_goal:
        if not st.session_state.api_ok:
            st.error("Add your GEMINI_API_KEY to .env or paste it in the sidebar.")
        else:
            with st.spinner("🧠 Analyzing…"):
                agent  = get_agent()
                result = agent.run(run_goal)
                st.session_state.history.insert(0, result)
                st.session_state.clear_goal_pending = True 
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

                    if res.get("output_file") and Path(res["output_file"]).exists():
                        out_path = Path(res["output_file"])
                        if out_path.suffix == ".csv":
                            out_df = pd.read_csv(out_path)
                            if len(out_df) > 1:
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