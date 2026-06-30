"""
main.py – CSV Insight Analyzer (smolagents-powered)
----------------------------------------------------
Architecture:
  PlannerAgent  (LiteLLMModel)  →  generates analysis plan
  CodeAgent     (LiteLLMModel)  →  writes + executes pandas code via smolagents sandbox
  InterpretAgent (LiteLLMModel) →  summarises results in plain English

Phoenix tracing is auto-enabled if a server is running on localhost:6006.
"""

import os, re, io, datetime, traceback, json, threading
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY and not GROQ_API_KEY:
    raise EnvironmentError("No API key found. Add GEMINI_API_KEY or GROQ_API_KEY to .env")

GEMINI_MODEL = "gemini/gemini-2.0-flash"
GROQ_MODEL   = "groq/llama-3.3-70b-versatile"

# ── Phoenix tracing (optional) ───────────────────────────────────────────────
def _setup_tracing():
    try:
        from phoenix.otel import register
        register(
            project_name="csv-insight-analyzer",
            endpoint="http://localhost:6006/v1/traces",
            auto_instrument=True,
        )
        print("✅ Phoenix tracing active → http://localhost:6006")
    except Exception:
        pass

_setup_tracing()

# ── smolagents + litellm ─────────────────────────────────────────────────────
from smolagents import CodeAgent, LiteLLMModel, tool
from litellm import completion
from litellm import RateLimitError as LiteLLMRateLimit

TEMPERATURE = 0.1

# Runtime fallback state
_gemini_healthy = bool(GEMINI_API_KEY)

print(f"🔵 Primary : {GEMINI_MODEL if GEMINI_API_KEY else GROQ_MODEL}")
if GROQ_API_KEY:
    print(f"🟢 Fallback: {GROQ_MODEL}")

# ── Directory setup ───────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
LOG_DIR    = ROOT / "logs"
CODE_DIR   = ROOT / "generated_code"
OUTPUT_DIR = ROOT / "analysis_output"
for d in (LOG_DIR, CODE_DIR, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

SUMMARY_CSV = LOG_DIR / "analysis_summary_log.csv"
MAX_RETRIES = 3


# ── Model factory with Gemini→Groq fallback ───────────────────────────────────
def _make_model() -> LiteLLMModel:
    """Return a LiteLLMModel pointing at the currently active provider."""
    global _gemini_healthy
    if _gemini_healthy and GEMINI_API_KEY:
        return LiteLLMModel(
            model_id=GEMINI_MODEL,
            api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
        )
    if GROQ_API_KEY:
        return LiteLLMModel(
            model_id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=TEMPERATURE,
        )
    # Last resort – Gemini even if unhealthy
    return LiteLLMModel(
        model_id=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE,
    )


def _llm_call(messages: list, temperature: float = TEMPERATURE) -> str:
    """
    Raw litellm call used for Plan and Interpret steps (not CodeAgent).
    Auto-falls back from Gemini to Groq on 429.
    """
    global _gemini_healthy

    model   = GEMINI_MODEL if (_gemini_healthy and GEMINI_API_KEY) else GROQ_MODEL
    api_key = GEMINI_API_KEY if (_gemini_healthy and GEMINI_API_KEY) else GROQ_API_KEY

    try:
        resp = completion(model=model, messages=messages,
                          api_key=api_key, temperature=temperature)
        return resp.choices[0].message.content

    except LiteLLMRateLimit:
        if model == GEMINI_MODEL and GROQ_API_KEY:
            print("⚠️  Gemini quota hit — switching to Groq for this session.")
            _gemini_healthy = False
            resp = completion(model=GROQ_MODEL, messages=messages,
                              api_key=GROQ_API_KEY, temperature=temperature)
            return resp.choices[0].message.content
        raise


def _strip_fences(text: str) -> str:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _append_summary(data: dict):
    df = pd.DataFrame([data])
    if SUMMARY_CSV.exists():
        df.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(SUMMARY_CSV, index=False)


# ── smolagents tools ──────────────────────────────────────────────────────────
# These are registered on the CodeAgent so it can call them during its loop.

_REGISTRY: Dict[str, pd.DataFrame] = {}   # populated per-run

@tool
def get_dataframe_schema() -> str:
    """
    Returns schema metadata for all loaded DataFrames including column names,
    dtypes, and unique values for low-cardinality categorical columns.
    Use this to understand the data structure before writing analysis code.
    """
    if not _REGISTRY:
        return "No DataFrames loaded."
    lines = []
    for name, df in _REGISTRY.items():
        col_parts = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            base  = f"'{col}' ({dtype})"
            if dtype in ("object", "str", "string") or "object" in dtype:
                n_unique = df[col].nunique()
                if n_unique <= 20:
                    vals = df[col].dropna().unique().tolist()
                    base += f" [values: {', '.join(repr(v) for v in vals[:20])}]"
            col_parts.append(base)
        lines.append(f"DataFrame '{name}': {', '.join(col_parts)}")
    return "\n".join(lines)


@tool
def get_dataframe_sample(df_name: str, n_rows: int = 5) -> str:
    """
    Returns the first n_rows of a loaded DataFrame as a string table.
    Use this to understand actual data values before filtering or grouping.

    Args:
        df_name: Name of the DataFrame (e.g. 'data', 'data2')
        n_rows:  Number of rows to preview (default 5, max 20)
    """
    if df_name not in _REGISTRY:
        return f"DataFrame '{df_name}' not found. Available: {list(_REGISTRY.keys())}"
    n_rows = min(n_rows, 20)
    return _REGISTRY[df_name].head(n_rows).to_string()


@tool
def save_result(result_str: str, filename: str = "result.csv") -> str:
    """
    Save a CSV string to the analysis_output directory.

    Args:
        result_str: CSV-formatted string to save.
        filename:   Output filename (default: result.csv).
    Returns:
        Absolute path of the saved file.
    """
    out_path = OUTPUT_DIR / filename
    out_path.write_text(result_str, encoding="utf-8")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
class DataAnalysisAgent:
    """
    Three-stage pipeline powered by smolagents:

    Stage 1 – PlannerAgent  : LiteLLMModel → numbered analysis plan
    Stage 2 – CodeAgent     : smolagents CodeAgent with tools → writes + runs pandas code
    Stage 3 – InterpretAgent: LiteLLMModel → plain-English insight from results
    """

    def __init__(self, all_dfs: Dict[str, pd.DataFrame], metadata: Dict[str, dict]):
        self.all_dfs  = all_dfs
        self.metadata = metadata
        self._ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = LOG_DIR / f"log_{self._ts}.txt"
        self._log_fh   = open(self._log_path, "w", encoding="utf-8")

        # Populate global registry so tools can access DataFrames
        global _REGISTRY
        _REGISTRY.clear()
        _REGISTRY.update({k: df.copy() for k, df in all_dfs.items()})

    def __del__(self):
        if hasattr(self, "_log_fh") and not self._log_fh.closed:
            self._log_fh.close()

    def _log(self, *args):
        msg = " ".join(map(str, args))
        print(msg)
        if not self._log_fh.closed:
            self._log_fh.write(msg + "\n")
            self._log_fh.flush()

    # ── Schema for prompts ────────────────────────────────────────────────────
    def _schema_str(self) -> str:
        lines = []
        for name, df in self.all_dfs.items():
            col_parts = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                base  = f"'{col}' ({dtype})"
                if dtype in ("object", "str", "string") or "object" in dtype:
                    n_unique = df[col].nunique()
                    if n_unique <= 20:
                        vals     = df[col].dropna().unique().tolist()
                        vals_str = ", ".join(repr(v) for v in vals[:20])
                        base    += f" [values: {vals_str}]"
                col_parts.append(base)
            lines.append(f"  DataFrame '{name}': {', '.join(col_parts)}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 – Planner (raw LLM call, no agent overhead needed)
    # ─────────────────────────────────────────────────────────────────────────
    def generate_plan(self, goal: str) -> str:
        prompt = (
            f"You are a senior data analyst. Given the schemas below and the goal, "
            f"write a clear numbered plan (no code).\n\n"
            f"Schemas:\n{self._schema_str()}\n\n"
            f"Goal: {goal}\n\n"
            f"Rules:\n"
            f"1. Only use listed column names.\n"
            f"2. Reference DataFrame names explicitly.\n"
            f"3. No code, no assumptions about data values.\n"
            f"4. Use exact string values shown in [values: ...] for any filtering steps.\n\n"
            f"Plan:"
        )
        self._log("\n─── Stage 1: Plan Prompt ───")
        self._log(prompt)
        plan = _llm_call([{"role": "user", "content": prompt}])
        self._log("\n─── Plan ───\n", plan)
        return plan

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 – CodeAgent (smolagents handles code generation + execution)
    # ─────────────────────────────────────────────────────────────────────────
    def _is_tabular_goal(self, goal: str) -> bool:
        """Heuristic: does this goal want a table/list, or just a number/answer?"""
        tabular_keywords = [
            "top", "list", "show", "give", "table", "rank", "breakdown",
            "details", "who are", "names of", "employees", "operators",
        ]
        scalar_keywords = [
            "how many", "count", "total", "average", "mean", "percentage",
            "what is", "what are", "maximum", "minimum", "max", "min",
        ]
        goal_lower = goal.lower()
        has_tabular = any(k in goal_lower for k in tabular_keywords)
        has_scalar  = any(k in goal_lower for k in scalar_keywords)
        # If both, tabular wins (e.g. "how many + top 10")
        return has_tabular

    def _build_agent_prompt(self, goal: str, plan: str) -> str:
        first_name   = next(iter(self.all_dfs))
        schema       = self._schema_str()
        is_tabular   = self._is_tabular_goal(goal)
        save_rule    = (
            "- For a single number or short text answer: just assign it to `result`. Do NOT save a CSV."
        ) if not is_tabular else (
            "- For a table result: assign it to `result` AND call the save_result tool:\n"
            "    save_result(result.to_csv(index=False), 'result.csv')"
        )

        return f"""You are an expert Python data analyst. Complete this task in as few steps as possible.

GOAL: {goal}

PLAN:
{plan}

DATA SCHEMA (use exact column names and values shown):
{schema}

ACCESS DATA:
  import pandas as pd, numpy as np
  df = all_dfs['{first_name}']   # keys available: {list(self.all_dfs.keys())}

RULES:
- Write ALL the pandas logic in a SINGLE code block on your FIRST step.
- Use EXACT string values from [values: ...] when filtering.
- Assign final answer to a variable named exactly `result`.
- {save_rule}
- DO NOT use os.path.join, os.makedirs, or import os — use the save_result tool instead.
- NO try/except blocks.
- Do NOT call get_dataframe_schema or get_dataframe_sample — the schema is already above.
- CRITICAL: After computing `result`, immediately call final_answer(result) in the SAME code block.
  Do NOT wait for another step. Do NOT rewrite the script again — call final_answer() and stop.
"""

    def _extract_from_memory(self, agent) -> tuple:
        """
        Extract execution logs + printed output from all completed steps.
        Returns (combined_text, code_snippets_list).
        Used to recover results even when the agent is interrupted mid-run.
        """
        logs, codes = [], []
        if not (hasattr(agent, "memory") and agent.memory.steps):
            return "", []
        for step in agent.memory.steps:
            # Collect code blocks (tool_calls may be None if the step errored
            # before any tool call was generated, e.g. a model/network failure)
            tool_calls = getattr(step, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    args = getattr(tc, "arguments", {})
                    code = args.get("code", "") if isinstance(args, dict) else str(args)
                    if code:
                        codes.append(code)
            # Collect execution output (printed results)
            if hasattr(step, "observations"):
                obs = step.observations or ""
                if obs:
                    logs.append(str(obs))
            # Also grab step output (use is not None — output may be a DataFrame)
            action_out = getattr(step, "action_output", None)
            if action_out is not None:
                try:
                    logs.append(str(action_out))
                except Exception:
                    pass
        return "\n".join(logs), codes

    def _run_code_agent(self, goal: str, plan: str) -> dict:
        """
        Runs the smolagents CodeAgent. It will:
        1. Call get_dataframe_schema() / get_dataframe_sample() as needed
        2. Write + execute pandas code in steps
        3. Self-correct on errors (up to MAX_RETRIES internally)
        Even if interrupted, extracts results from completed memory steps.
        """
        global _gemini_healthy

        prompt = self._build_agent_prompt(goal, plan)
        self._log("\n─── Stage 2: CodeAgent Prompt ───")
        self._log(prompt)

        additional_imports = [
            "pandas", "numpy", "os", "json",
            "matplotlib", "seaborn", "datetime", "re",
        ]

        AGENT_TIMEOUT_SECONDS = 60   # hard ceiling per attempt

        for attempt in range(1, MAX_RETRIES + 1):
            self._log(f"\n─── CodeAgent Attempt {attempt}/{MAX_RETRIES} ───")
            agent = None
            try:
                model = _make_model()
                agent = CodeAgent(
                    model=model,
                    tools=[get_dataframe_schema, get_dataframe_sample, save_result],
                    additional_authorized_imports=additional_imports,
                    max_steps=3,   # Single-step completion enforced via prompt; 3 is a safety ceiling
                )

                # ── Run with a hard timeout (thread-based, Windows-safe) ───
                result_holder: dict = {}
                error_holder:  dict = {}

                def _run_agent():
                    try:
                        result_holder["value"] = agent.run(
                            prompt,
                            additional_args={
                                "all_dfs":    {k: df.copy() for k, df in self.all_dfs.items()},
                                "output_dir": str(OUTPUT_DIR),
                            },
                        )
                    except Exception as exc:
                        error_holder["error"] = exc

                worker = threading.Thread(target=_run_agent, daemon=True)
                worker.start()
                worker.join(timeout=AGENT_TIMEOUT_SECONDS)

                if worker.is_alive():
                    # Timed out — try to recover from whatever steps completed
                    self._log(f"⏱ CodeAgent exceeded {AGENT_TIMEOUT_SECONDS}s — recovering from partial progress.")
                    exec_logs, code_snippets = self._extract_from_memory(agent)
                    if exec_logs:
                        code_path = CODE_DIR / f"code_{self._ts}_attempt{attempt}.py"
                        if code_snippets:
                            code_path.write_text("\n\n".join(code_snippets), encoding="utf-8")
                        output_file = self._find_output_file()
                        return {
                            "status":      "success",
                            "result":      exec_logs,
                            "output_file": output_file,
                            "code_path":   str(code_path),
                        }
                    return {"status": "failed", "error": f"Timed out after {AGENT_TIMEOUT_SECONDS}s with no recoverable output."}

                if "error" in error_holder:
                    raise error_holder["error"]

                agent_result = result_holder.get("value")
                self._log(f"\n─── CodeAgent Result ───\n{agent_result}")

            except Exception as e:
                err_str = str(e)
                self._log(f"CodeAgent interrupted: {err_str[:200]}")

                # ── Recovery: extract results from completed steps ────────
                if agent is not None:
                    exec_logs, code_snippets = self._extract_from_memory(agent)
                    if exec_logs:
                        self._log("✅ Recovering from completed steps...")
                        # Save extracted code
                        code_path = CODE_DIR / f"code_{self._ts}_attempt{attempt}.py"
                        if code_snippets:
                            code_path.write_text("\n\n".join(code_snippets), encoding="utf-8")
                        output_file = self._find_output_file()
                        return {
                            "status":      "success",
                            "result":      exec_logs,   # Use printed output as result
                            "output_file": output_file,
                            "code_path":   str(code_path),
                        }

                # If rate limit, switch to Groq and retry
                if "429" in err_str or "RateLimit" in err_str or "rate_limit" in err_str.lower():
                    if _gemini_healthy and GROQ_API_KEY:
                        print("⚠️  Gemini quota hit — switching to Groq for this session.")
                    _gemini_healthy = False
                    continue

                # Transient network/server hiccup (e.g. Groq "Server disconnected") — just retry
                transient_markers = [
                    "Server disconnected", "RemoteProtocolError",
                    "InternalServerError", "Connection reset",
                ]
                if any(m in err_str for m in transient_markers):
                    self._log(f"🔁 Transient network error — retrying attempt {attempt + 1}.")
                    continue

                # Non-rate-limit error with no recoverable memory
                if attempt == MAX_RETRIES:
                    return {"status": "failed", "error": err_str}
                continue

            # ── Successful run: save code and return ──────────────────────
            code_path = CODE_DIR / f"code_{self._ts}_attempt{attempt}.py"
            _, code_snippets = self._extract_from_memory(agent)
            if code_snippets:
                code_path.write_text("\n\n".join(code_snippets), encoding="utf-8")

            output_file = self._find_output_file()
            return {
                "status":      "success",
                "result":      agent_result,
                "output_file": output_file,
                "code_path":   str(code_path),
            }

        return {"status": "failed", "error": "All CodeAgent attempts exhausted."}

    def _find_output_file(self) -> Optional[str]:
        """Return the most recently written CSV in OUTPUT_DIR."""
        csvs = sorted(OUTPUT_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(csvs[0]) if csvs else None

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 – Interpreter (raw LLM call)
    # ─────────────────────────────────────────────────────────────────────────
    def interpret(self, goal: str, agent_result: Any, output_file: Optional[str]) -> str:
        parts = [f"Analysis goal: '{goal}'\n\n"]

        if agent_result is not None:
            if isinstance(agent_result, pd.DataFrame):
                display = agent_result if len(agent_result) <= 50 else agent_result.head(50)
                parts.append(
                    f"Result DataFrame ({len(agent_result)} rows):\n"
                    f"{display.to_string(index=False)}\n"
                )
            elif isinstance(agent_result, pd.Series):
                display = agent_result if len(agent_result) <= 50 else agent_result.head(50)
                parts.append(f"Result Series:\n{display.to_string()}\n")
            else:
                parts.append(f"Agent result: {agent_result}\n")

        if output_file and Path(output_file).exists() and output_file.endswith(".csv"):
            try:
                out_df  = pd.read_csv(output_file)
                display = out_df if len(out_df) <= 50 else out_df.head(50)
                parts.append(
                    f"\nOutput CSV ({len(out_df)} rows):\n"
                    f"{display.to_string(index=False)}\n"
                )
            except Exception:
                pass

        parts.append(
            "\nIMPORTANT: Write a clear insight summary using ONLY the exact numbers "
            "and values shown above. Do NOT use placeholders like [insert number]. "
            "Do NOT invent or estimate values. No code, no file paths in the response."
        )

        prompt = "".join(parts)
        self._log("\n─── Stage 3: Interpretation Prompt ───\n", prompt)
        return _llm_call([{"role": "user", "content": prompt}], temperature=0.1)

    def _is_simple_question(self, goal: str) -> bool:
        """Simple count/scalar questions don't need a separate Plan LLM call."""
        simple_patterns = [
            "how many", "count", "what is the total", "what is the average",
            "what is the max", "what is the min", "percentage of",
        ]
        g = goal.lower().strip()
        # Simple if it matches a simple pattern AND has no "and also" / compound asks
        is_simple = any(g.startswith(p) or g.startswith("how many") for p in simple_patterns)
        is_compound = any(w in g for w in [" and ", " also ", " plus ", "top ", "list ", "show "])
        return is_simple and not is_compound

    # ─────────────────────────────────────────────────────────────────────────
    # Orchestrator
    # ─────────────────────────────────────────────────────────────────────────
    def run(self, goal: str) -> dict:
        summary = {
            "timestamp":      self._ts,
            "goal":           goal,
            "status":         "Failed",
            "error":          "",
            "code_path":      "",
            "output_file":    "",
            "interpretation": "",
            "log_path":       str(self._log_path),
        }

        # Stage 1 – Plan (skip for simple scalar questions to save time)
        if self._is_simple_question(goal):
            self._log("⚡ Simple question detected — skipping Plan stage.")
            plan = f"Write pandas code to answer: {goal}"
        else:
            try:
                plan = self.generate_plan(goal)
                if not plan.strip():
                    raise ValueError("Empty plan returned.")
            except Exception as e:
                summary["error"] = f"Plan failed: {e}"
                _append_summary(summary)
                return summary

        # Stage 2 – CodeAgent
        exec_result = self._run_code_agent(goal, plan)
        summary["code_path"] = exec_result.get("code_path", "")

        if exec_result["status"] != "success":
            summary["error"] = exec_result.get("error", "Unknown error")
            _append_summary(summary)
            return summary

        # Stage 3 – Interpret
        summary["output_file"]    = exec_result.get("output_file") or ""
        summary["interpretation"] = self.interpret(
            goal,
            exec_result.get("result"),
            exec_result.get("output_file"),
        )
        summary["status"] = "Success"
        _append_summary(summary)
        return summary


# ── CLI entrypoint ────────────────────────────────────────────────────────────
def load_csvs_cli() -> tuple[Dict, Dict]:
    all_dfs, metadata = {}, {}
    print("Load up to 3 CSV files (blank to finish).")
    for i in range(1, 4):
        path = input(f"CSV {i} path: ").strip().strip('"')
        if not path:
            break
        p = Path(path).expanduser()
        if p.exists() and p.suffix.lower() == ".csv":
            key = "data" if i == 1 else f"data{i}"
            df  = pd.read_csv(p)
            all_dfs[key]  = df
            metadata[key] = {c: str(t) for c, t in df.dtypes.items()}
            print(f"✅ '{p.name}' → '{key}' ({len(df)} rows, {len(df.columns)} cols)")
        else:
            print("❌ Not found or not a CSV – skipped.")
    return all_dfs, metadata


if __name__ == "__main__":
    all_dfs, metadata = load_csvs_cli()
    if not all_dfs:
        print("No CSVs loaded. Exiting.")
        raise SystemExit

    agent = DataAnalysisAgent(all_dfs, metadata)
    while True:
        goal = input("\nAnalysis goal (or 'exit'): ").strip()
        if goal.lower() in {"exit", "quit", "q"}:
            print("👋 Goodbye!")
            break
        if not goal:
            continue
        result = agent.run(goal)
        if result["status"] == "Success":
            print("\n✅ Done!")
            print("📊 Insight:\n", result["interpretation"])
            if result["output_file"]:
                print("💾 Output:", result["output_file"])
        else:
            print("\n❌ Failed:", result["error"])
        agent = DataAnalysisAgent(all_dfs, metadata)