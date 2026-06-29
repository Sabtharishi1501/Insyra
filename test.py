"""
test.py – Integration tests for DataAnalysisAgent
---------------------------------------------------
Run:  python test.py
Does NOT call the real LLM – patches _llm() with deterministic fakes
so tests are fast, free, and offline.
"""

import os, sys, io, json, unittest, tempfile, textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

# ── Ensure local main.py is importable ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Fake API key so main.py doesn't raise on import ─────────────────────────
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol", "Dave"],
        "dept":   ["Eng", "HR", "Eng", "Finance"],
        "salary": [90000, 55000, 95000, 72000],
        "score":  [88, 72, 95, 80],
    })


def _make_agent(df: pd.DataFrame = None):
    from main import DataAnalysisAgent
    if df is None:
        df = _sample_df()
    meta = {"data": {c: str(t) for c, t in df.dtypes.items()}}
    return DataAnalysisAgent({"data": df}, meta)


FAKE_PLAN = "1. Load data.\n2. Group by dept.\n3. Compute mean salary.\n4. Assign result."

FAKE_CODE = textwrap.dedent("""
    import pandas as pd, os
    df = all_dfs['data']
    df_result = df.groupby('dept')['salary'].mean().reset_index()
    df_result.columns = ['dept', 'avg_salary']
    os.makedirs(output_dir, exist_ok=True)
    df_result.to_csv(os.path.join(output_dir, 'result.csv'), index=False)
    result = df_result
""").strip()

FAKE_INTERP = "Engineering has the highest average salary at $92,500."


def _fake_llm(messages, temperature=0.1):
    """Returns plan, code, or interpretation based on prompt content."""
    content = messages[0]["content"]
    if "List clear action steps" in content or "write a clear numbered plan" in content.lower():
        return FAKE_PLAN
    if "CODE REQUIREMENTS" in content or "expert Python data analyst" in content:
        return f"```python\n{FAKE_CODE}\n```"
    return FAKE_INTERP


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────
class TestSchemaString(unittest.TestCase):
    def test_schema_contains_all_columns(self):
        agent = _make_agent()
        schema = agent._schema_str()
        for col in ("name", "dept", "salary", "score"):
            self.assertIn(col, schema)

    def test_schema_contains_dtypes(self):
        agent = _make_agent()
        schema = agent._schema_str()
        # schema converts dtypes via str(), so 'object' becomes 'str' in some pandas versions
        self.assertTrue("int64" in schema or "int" in schema)
        self.assertTrue("str" in schema or "object" in schema)


class TestStripFences(unittest.TestCase):
    def test_strips_python_fence(self):
        from main import _strip_fences
        raw = "```python\nprint('hi')\n```"
        self.assertEqual(_strip_fences(raw), "print('hi')")

    def test_passthrough_bare_code(self):
        from main import _strip_fences
        code = "x = 1 + 1"
        self.assertEqual(_strip_fences(code), code)


class TestPlanGeneration(unittest.TestCase):
    @patch("main._llm", side_effect=_fake_llm)
    def test_plan_non_empty(self, _):
        agent = _make_agent()
        plan  = agent.generate_plan("Average salary by department")
        self.assertIsInstance(plan, str)
        self.assertGreater(len(plan.strip()), 0)

    @patch("main._llm", return_value="")
    def test_empty_plan_returns_failed(self, _):
        agent  = _make_agent()
        result = agent.run("does not matter")
        self.assertEqual(result["status"], "Failed")
        self.assertIn("Plan", result["error"])


class TestCodeExecution(unittest.TestCase):
    def test_successful_exec(self):
        agent = _make_agent()
        res   = agent._execute(FAKE_CODE, attempt=1)
        self.assertEqual(res["status"], "success")
        self.assertIsInstance(res["result"], pd.DataFrame)
        self.assertIn("dept", res["result"].columns)

    def test_failed_exec_returns_error(self):
        agent = _make_agent()
        bad   = "raise ValueError('intentional test error')"
        res   = agent._execute(bad, attempt=1)
        self.assertEqual(res["status"], "failed")
        self.assertIn("intentional test error", res["error"])

    def test_result_variable_captured(self):
        agent  = _make_agent()
        code   = "result = 42"
        res    = agent._execute(code, attempt=1)
        self.assertEqual(res["result"], 42)

    def test_dataframe_copy_isolation(self):
        """Exec should not mutate the original DataFrame."""
        df    = _sample_df()
        agent = _make_agent(df)
        mutating = "all_dfs['data']['new_col'] = 999"
        agent._execute(mutating, attempt=1)
        self.assertNotIn("new_col", df.columns)


class TestInterpretation(unittest.TestCase):
    @patch("main._llm", return_value=FAKE_INTERP)
    def test_interpret_dataframe(self, _):
        agent  = _make_agent()
        result = _sample_df()
        text   = agent.interpret("avg salary by dept", result, None)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    @patch("main._llm", return_value=FAKE_INTERP)
    def test_interpret_scalar(self, _):
        agent = _make_agent()
        text  = agent.interpret("count rows", 4, None)
        self.assertIsInstance(text, str)


class TestFullRun(unittest.TestCase):
    @patch("main._llm", side_effect=_fake_llm)
    def test_success_run(self, _):
        agent  = _make_agent()
        result = agent.run("Average salary by department")
        self.assertEqual(result["status"], "Success")
        self.assertIn("interpretation", result)
        self.assertGreater(len(result["interpretation"]), 0)

    @patch("main._llm", return_value="")
    def test_plan_failure_returns_failed_status(self, _):
        agent  = _make_agent()
        result = agent.run("some goal")
        self.assertEqual(result["status"], "Failed")
        self.assertIn("error", result)

    @patch("main._llm", side_effect=_fake_llm)
    def test_summary_dict_keys_present(self, _):
        agent    = _make_agent()
        result   = agent.run("Average salary by department")
        required = {"timestamp", "goal", "status", "error", "code_path",
                    "output_file", "interpretation", "log_path"}
        self.assertTrue(required.issubset(result.keys()))


class TestRetryLoop(unittest.TestCase):
    """Agent should retry up to MAX_RETRIES then return Failed."""

    @patch("main._llm", side_effect=_fake_llm)
    def test_retry_on_bad_code(self, mock_llm):
        """First two code attempts are bad; third is valid."""
        call_count = {"n": 0}

        def selective_llm(messages, temperature=0.1):
            content = messages[0]["content"]
            # Plan call
            if "write a clear numbered plan" in content.lower() or "List clear action steps" in content:
                return FAKE_PLAN
            # First two code calls → garbage; third → real code
            call_count["n"] += 1
            if call_count["n"] < 3:
                return "```python\nraise RuntimeError('simulated failure')\n```"
            return f"```python\n{FAKE_CODE}\n```"

        with patch("main._llm", side_effect=selective_llm):
            agent  = _make_agent()
            result = agent.run("avg salary by dept")
        self.assertEqual(result["status"], "Success")

    @patch("main._llm", return_value=f"```python\nraise RuntimeError('always fails')\n```")
    def test_all_retries_exhausted(self, _):
        # Plan must succeed for the retry loop to start
        def llm_patch(messages, temperature=0.1):
            content = messages[0]["content"]
            if "write a clear numbered plan" in content.lower() or "List clear action steps" in content:
                return FAKE_PLAN
            return "```python\nraise RuntimeError('always fails')\n```"

        with patch("main._llm", side_effect=llm_patch):
            agent  = _make_agent()
            result = agent.run("will always fail")
        self.assertEqual(result["status"], "Failed")


class TestOutputFileSave(unittest.TestCase):
    @patch("main._llm", side_effect=_fake_llm)
    def test_result_csv_created(self, _):
        agent  = _make_agent()
        result = agent.run("Average salary by department")
        if result.get("output_file"):
            self.assertTrue(Path(result["output_file"]).exists())


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CSV Insight Analyzer — Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=".", pattern="test*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)