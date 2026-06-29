"""
main.py – DataAnalysisAgent core
---------------------------------
• Loads API key from .env via python-dotenv
• Real exec() sandbox with error-retry loop (up to 3 attempts)
• Plan → Code → Execute → Interpret pipeline
• Phoenix OTEL tracing (optional, skipped if server not running)
• No dead globals, no passthrough tools
"""

import os, re, io, datetime, traceback, json
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env first ──────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY and not GROQ_API_KEY:
    raise EnvironmentError("No API key found. Add GEMINI_API_KEY or GROQ_API_KEY to .env")

GEMINI_MODEL = "gemini/gemini-2.0-flash"
GROQ_MODEL   = "groq/llama-3.3-70b-versatile"

# ── Optional Phoenix tracing ──────────────────────────────────────────────────
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
        pass  # Phoenix server not running – skip silently

_setup_tracing()

# ── litellm ──────────────────────────────────────────────────────────────────
from litellm import completion
from litellm import RateLimitError as LiteLLMRateLimit

TEMPERATURE = 0.1

# Runtime state – tracks which model is active this session
_active_model   = GEMINI_MODEL if GEMINI_API_KEY else GROQ_MODEL
_active_key     = GEMINI_API_KEY if GEMINI_API_KEY else GROQ_API_KEY
_gemini_healthy = bool(GEMINI_API_KEY)   # flips to False on 429

print(f"🔵 Primary model: {_active_model}")
if GROQ_API_KEY:
    print(f"🟢 Fallback model: {GROQ_MODEL}")

# ── Directory setup ───────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
LOG_DIR    = ROOT / "logs"
CODE_DIR   = ROOT / "generated_code"
OUTPUT_DIR = ROOT / "analysis_output"
for d in (LOG_DIR, CODE_DIR, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

SUMMARY_CSV = LOG_DIR / "analysis_summary_log.csv"

MAX_RETRIES = 3


# ─────────────────────────────────────────────────────────────────────────────
def _current_model() -> tuple[str, str]:
    """Return (model_id, api_key) for the currently active provider."""
    global _active_model, _active_key, _gemini_healthy
    if _gemini_healthy and GEMINI_API_KEY:
        return GEMINI_MODEL, GEMINI_API_KEY
    if GROQ_API_KEY:
        return GROQ_MODEL, GROQ_API_KEY
    # Gemini is the only option even if unhealthy
    return GEMINI_MODEL, GEMINI_API_KEY


def _llm(messages: list, temperature: float = TEMPERATURE) -> str:
    """
    Single LLM helper with automatic Gemini → Groq fallback.
    - Tries Gemini first (if key available and not rate-limited).
    - On 429 RateLimitError, switches to Groq for the rest of the session.
    - Raises if both providers fail.
    """
    global _gemini_healthy, _active_model, _active_key

    model, api_key = _current_model()

    try:
        resp = completion(
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    except LiteLLMRateLimit as e:
        # Gemini quota exceeded – flip to Groq if available
        if model == GEMINI_MODEL and GROQ_API_KEY:
            print("⚠️  Gemini quota hit — switching to Groq for this session.")
            _gemini_healthy = False
            _active_model   = GROQ_MODEL
            _active_key     = GROQ_API_KEY
            # Retry immediately with Groq
            resp = completion(
                model=GROQ_MODEL,
                messages=messages,
                api_key=GROQ_API_KEY,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        raise  # No fallback available

    except Exception:
        raise


def _strip_fences(text: str) -> str:
    """Remove ```python … ``` fences; return bare code."""
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _append_summary(data: dict):
    df = pd.DataFrame([data])
    if SUMMARY_CSV.exists():
        df.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(SUMMARY_CSV, index=False)


# ─────────────────────────────────────────────────────────────────────────────
class DataAnalysisAgent:
    """
    Orchestrates: plan → code-gen → real exec() sandbox → interpretation.
    Each run() creates its own timestamped log file.
    """

    def __init__(self, all_dfs: Dict[str, pd.DataFrame], metadata: Dict[str, dict]):
        self.all_dfs   = all_dfs
        self.metadata  = metadata          # {df_name: {col: dtype_str}}
        self._ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = LOG_DIR / f"log_{self._ts}.txt"
        self._log_fh   = open(self._log_path, "w", encoding="utf-8")

    def __del__(self):
        if hasattr(self, "_log_fh") and not self._log_fh.closed:
            self._log_fh.close()

    # ── Logging ──────────────────────────────────────────────────────────────
    def _log(self, *args):
        msg = " ".join(map(str, args))
        print(msg)
        if not self._log_fh.closed:
            self._log_fh.write(msg + "\n")
            self._log_fh.flush()

    # ── Schema string ────────────────────────────────────────────────────────
    def _schema_str(self) -> str:
        """
        Build schema string. For object/string columns with <=20 unique values,
        include the actual unique values so the LLM knows exact filter strings.
        """
        lines = []
        for name, cols in self.metadata.items():
            col_parts = []
            df = self.all_dfs[name]
            for col, dtype in cols.items():
                base = f"'{col}' ({dtype})"
                # Show unique values for low-cardinality categorical columns
                if dtype in ("object", "str", "string") or "object" in dtype:
                    n_unique = df[col].nunique()
                    if n_unique <= 20:
                        unique_vals = df[col].dropna().unique().tolist()
                        vals_str = ", ".join(f'"{v}"' for v in unique_vals[:20])
                        base += f" [values: {vals_str}]"
                col_parts.append(base)
            col_list = ", ".join(col_parts)
            lines.append(f"  DataFrame '{name}': {col_list}")
        return "\n".join(lines)

    # ── Step 1: Plan ─────────────────────────────────────────────────────────
    def generate_plan(self, goal: str) -> str:
        prompt = (
            f"You are a senior data analyst. Given the schemas below and the goal, "
            f"write a clear numbered plan (no code).\n\n"
            f"Schemas:\n{self._schema_str()}\n\n"
            f"Goal: {goal}\n\n"
            f"Rules:\n"
            f"1. Only use listed column names.\n"
            f"2. Reference DataFrame names explicitly.\n"
            f"3. No code, no assumptions about data values.\n\n"
            f"Plan:"
        )
        self._log("\n─── Plan Prompt ───")
        self._log(prompt)
        plan = _llm([{"role": "user", "content": prompt}])
        self._log("\n─── Plan ───\n", plan)
        return plan

    # ── Step 2: Code generation ───────────────────────────────────────────────
    def _build_code_prompt(self, goal: str, plan: str, prev_error: str = "") -> str:
        first_name = next(iter(self.metadata))
        cols = self._schema_str()

        prompt = f"""You are an expert Python data analyst.

GOAL: {goal}

PLAN (follow this):
{plan}

DATA SCHEMA:
{cols}

CODE REQUIREMENTS:
- All DataFrames are already loaded in a dict called `all_dfs`.
  Access them as: df = all_dfs['{first_name}']
- Do NOT call pd.read_csv().
- Import pandas as pd, numpy as np, os at the top.
- Save the final DataFrame/Series result:
    df_result.to_csv(os.path.join(output_dir, 'result.csv'), index=False)
- Assign the key finding to a variable named exactly `result`.
- NO try/except blocks.
- Return ONLY a python code block, no prose.

CRITICAL — STRING FILTERING RULES:
- The DATA SCHEMA above shows [values: ...] for categorical columns.
- ALWAYS use the EXACT values shown in [values: ...] when filtering strings.
- NEVER guess or assume string values like "dayshift", "nightshift", "Day", "Night".
- Use value_counts() to group categories — do NOT hardcode filter strings unless
  they exactly match a value shown in the schema [values: ...] list.
- For shift/category counting, prefer: df['col'].value_counts().reset_index()
"""
        if prev_error:
            prompt += f"\nPREVIOUS ERROR (fix this):\n```\n{prev_error}\n```\n"
        return prompt

    # ── Step 3: Real exec() sandbox ──────────────────────────────────────────
    def _execute(self, code: str, attempt: int) -> dict:
        code_path = CODE_DIR / f"code_{self._ts}_attempt{attempt}.py"
        code_path.write_text(code, encoding="utf-8")
        self._log(f"Code saved → {code_path}")

        g = {
            "all_dfs":    {k: df.copy() for k, df in self.all_dfs.items()},
            "pd":         pd,
            "np":         np,
            "os":         os,
            "output_dir": str(OUTPUT_DIR),
        }

        # Inject matplotlib only when the code uses it
        if "plt." in code:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                g["plt"] = plt
            except ImportError:
                pass

        try:
            exec(compile(code, str(code_path), "exec"), g)
        except Exception:
            err = traceback.format_exc()
            self._log("Exec error:\n", err)
            return {"status": "failed", "error": err}

        result_var  = g.get("result")
        output_file = self._find_output_file(code)
        return {
            "status":      "success",
            "result":      result_var,
            "output_file": output_file,
        }

    def _find_output_file(self, code: str) -> Optional[str]:
        patterns = [
            r"\.to_csv\(['\"]?[^'\"]*?([a-zA-Z0-9_.-]+\.csv)",
            r"\.to_excel\(['\"]?[^'\"]*?([a-zA-Z0-9_.-]+\.xlsx?)",
            r"plt\.savefig\(['\"]?[^'\"]*?([a-zA-Z0-9_.-]+\.(?:png|jpg|pdf))",
        ]
        for pat in patterns:
            m = re.search(pat, code)
            if m:
                candidate = OUTPUT_DIR / m.group(1).strip("'\"")
                if candidate.exists():
                    return str(candidate)
        return None

    # ── Step 4: Interpret ────────────────────────────────────────────────────
    def interpret(self, goal: str, result_var: Any, output_file: Optional[str]) -> str:
        parts = [f"Analysis goal: '{goal}'\n\n"]

        if result_var is not None:
            if isinstance(result_var, pd.DataFrame):
                # Send ALL rows if small, else first 50 — no placeholders
                display_rows = result_var if len(result_var) <= 50 else result_var.head(50)
                parts += [
                    f"Result type: DataFrame ({len(result_var)} rows x {len(result_var.columns)} cols)\n",
                    f"FULL DATA (use these exact numbers in your summary):\n{display_rows.to_string(index=False)}\n",
                ]
            elif isinstance(result_var, pd.Series):
                display_rows = result_var if len(result_var) <= 50 else result_var.head(50)
                parts += [
                    f"Result type: Series ({len(result_var)} entries)\n",
                    f"FULL DATA (use these exact numbers in your summary):\n{display_rows.to_string()}\n",
                ]
            else:
                parts.append(f"Result ({type(result_var).__name__}): {result_var}\n")

        # Also read full CSV output if available
        if output_file and Path(output_file).exists():
            if output_file.endswith(".csv"):
                try:
                    out_df = pd.read_csv(output_file)
                    display_rows = out_df if len(out_df) <= 50 else out_df.head(50)
                    parts.append(
                        f"\nOutput CSV ({len(out_df)} rows) — use these exact numbers:\n"
                        f"{display_rows.to_string(index=False)}\n"
                    )
                except Exception:
                    pass

        parts.append(
            "\nIMPORTANT: Write a clear insight summary using ONLY the exact numbers "
            "and values from the data above. Do NOT use placeholders like [insert number]. "
            "Do NOT invent or estimate any values. No code, no file paths."
        )
        prompt = "".join(parts)
        self._log("\n─── Interpretation Prompt ───\n", prompt)
        return _llm([{"role": "user", "content": prompt}], temperature=0.1)

    # ── Orchestrator ─────────────────────────────────────────────────────────
    def run(self, goal: str) -> dict:
        summary = {
            "timestamp":         self._ts,
            "goal":              goal,
            "status":            "Failed",
            "error":             "",
            "code_path":         "",
            "output_file":       "",
            "interpretation":    "",
            "log_path":          str(self._log_path),
        }

        # Plan
        try:
            plan = self.generate_plan(goal)
            if not plan.strip():
                raise ValueError("Empty plan returned.")
        except Exception as e:
            summary["error"] = f"Plan failed: {e}"
            _append_summary(summary)
            return summary

        # Code → exec loop
        prev_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            self._log(f"\n─── Attempt {attempt}/{MAX_RETRIES} ───")
            code_prompt = self._build_code_prompt(goal, plan, prev_error)
            self._log("\n─── Code Prompt ───\n", code_prompt)

            try:
                raw = _llm([{"role": "user", "content": code_prompt}])
                code = _strip_fences(raw)
                if not code:
                    raise ValueError("Empty code returned.")
                self._log("\n─── Generated Code ───\n", code)
            except Exception as e:
                prev_error = str(e)
                summary["error"] = f"Code-gen failed attempt {attempt}: {e}"
                continue

            exec_result = self._execute(code, attempt)
            summary["code_path"] = str(CODE_DIR / f"code_{self._ts}_attempt{attempt}.py")

            if exec_result["status"] == "success":
                summary["output_file"]    = exec_result.get("output_file") or ""
                summary["interpretation"] = self.interpret(
                    goal, exec_result["result"], exec_result.get("output_file")
                )
                summary["status"] = "Success"
                summary["error"]  = ""
                _append_summary(summary)
                return summary
            else:
                prev_error = exec_result["error"]
                summary["error"] = f"Exec failed attempt {attempt}: {prev_error}"
                self._log(f"Retrying due to: {prev_error[:200]}")

        self._log(f"All {MAX_RETRIES} attempts failed.")
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
            df = pd.read_csv(p)
            all_dfs[key]  = df
            metadata[key] = {c: str(t) for c, t in df.dtypes.items()}
            print(f"✅ '{p.name}' → '{key}' ({len(df)} rows, {len(df.columns)} cols)")
        else:
            print(f"❌ Not found or not a CSV – skipped.")
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

        # Fresh agent per run (new log file, fresh timestamp)
        agent = DataAnalysisAgent(all_dfs, metadata)