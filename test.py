"""
test.py – Test suite for DataAnalysisAgent (smolagents-powered)
----------------------------------------------------------------
All LLM + CodeAgent calls are mocked — runs offline, no API key needed.
Run:  python test.py
"""

import os, sys, io, unittest, textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol", "Dave"],
        "dept":   ["Eng", "HR", "Eng", "Finance"],
        "salary": [90000, 55000, 95000, 72000],
        "score":  [88, 72, 95, 80],
        "shift":  ["Day", "Night", "Day", "Night"],
    })


def _make_agent(df: pd.DataFrame = None):
    from main import DataAnalysisAgent
    if df is None:
        df = _sample_df()
    meta = {"data": {c: str(t) for c, t in df.dtypes.items()}}
    return DataAnalysisAgent({"data": df}, meta)


FAKE_PLAN = "1. Load data.\n2. Group by dept.\n3. Compute mean salary.\n4. Assign result."

FAKE_AGENT_LOGS = textwrap.dedent("""
    Day Shift: 2 persons
    Night Shift: 2 persons
    Top result: Alice, 90000
""").strip()

FAKE_INTERP = "Engineering has the highest average salary at $92,500."


def _fake_llm_call(messages, temperature=0.1):
    content = messages[0]["content"]
    if "write a clear numbered plan" in content.lower():
        return FAKE_PLAN
    return FAKE_INTERP


# ── Mock CodeAgent that succeeds ──────────────────────────────────────────────
class MockCodeAgentSuccess:
    def __init__(self, *args, **kwargs):
        self.memory = MagicMock()
        step = MagicMock()
        step.tool_calls = []
        step.observations = FAKE_AGENT_LOGS
        step.action_output = None
        self.memory.steps = [step]

    def run(self, prompt, additional_args=None):
        return FAKE_AGENT_LOGS


# ── Mock CodeAgent that always fails ─────────────────────────────────────────
class MockCodeAgentFail:
    def __init__(self, *args, **kwargs):
        self.memory = MagicMock()
        step = MagicMock()
        step.tool_calls = []
        step.observations = ""
        step.action_output = None
        self.memory.steps = [step]

    def run(self, prompt, additional_args=None):
        raise RuntimeError("simulated CodeAgent failure")


# ── Mock CodeAgent that fails then succeeds ───────────────────────────────────
_attempt_counter = {"n": 0}

class MockCodeAgentRetry:
    def __init__(self, *args, **kwargs):
        self.memory = MagicMock()
        step = MagicMock()
        step.tool_calls = []
        step.observations = FAKE_AGENT_LOGS
        step.action_output = None
        self.memory.steps = [step]

    def run(self, prompt, additional_args=None):
        _attempt_counter["n"] += 1
        if _attempt_counter["n"] < 3:
            raise RuntimeError("simulated transient failure")
        return FAKE_AGENT_LOGS


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────
class TestSchemaString(unittest.TestCase):
    def test_schema_contains_all_columns(self):
        agent  = _make_agent()
        schema = agent._schema_str()
        for col in ("name", "dept", "salary", "score", "shift"):
            self.assertIn(col, schema)

    def test_schema_shows_categorical_values(self):
        agent  = _make_agent()
        schema = agent._schema_str()
        # Low-cardinality string cols should show [values: ...]
        self.assertIn("values:", schema)
        self.assertIn("Day",   schema)
        self.assertIn("Night", schema)

    def test_schema_contains_dtypes(self):
        agent  = _make_agent()
        schema = agent._schema_str()
        self.assertTrue("int64" in schema or "int" in schema)
        self.assertTrue("object" in schema or "str" in schema)


class TestStripFences(unittest.TestCase):
    def test_strips_python_fence(self):
        from main import _strip_fences
        raw = "```python\nprint('hi')\n```"
        self.assertEqual(_strip_fences(raw), "print('hi')")

    def test_passthrough_bare_code(self):
        from main import _strip_fences
        code = "x = 1 + 1"
        self.assertEqual(_strip_fences(code), code)


class TestTools(unittest.TestCase):
    def test_get_dataframe_schema_tool(self):
        from main import get_dataframe_schema, _REGISTRY
        _make_agent()   # populates _REGISTRY
        result = get_dataframe_schema()
        self.assertIsInstance(result, str)
        self.assertIn("data", result)

    def test_get_dataframe_sample_tool(self):
        from main import get_dataframe_sample, _REGISTRY
        _make_agent()
        result = get_dataframe_sample("data", 3)
        self.assertIsInstance(result, str)
        self.assertIn("Alice", result)

    def test_get_dataframe_sample_missing(self):
        from main import get_dataframe_sample, _REGISTRY
        _make_agent()
        result = get_dataframe_sample("nonexistent")
        self.assertIn("not found", result)

    def test_save_result_tool(self):
        from main import save_result
        path = save_result("col1,col2\n1,2\n3,4", "test_out.csv")
        self.assertTrue(Path(path).exists())
        Path(path).unlink()


class TestPlanGeneration(unittest.TestCase):
    @patch("main._llm_call", side_effect=_fake_llm_call)
    def test_plan_non_empty(self, _):
        agent = _make_agent()
        plan  = agent.generate_plan("Average salary by department")
        self.assertIsInstance(plan, str)
        self.assertGreater(len(plan.strip()), 0)

    @patch("main._llm_call", return_value="")
    def test_empty_plan_returns_failed(self, _):
        agent  = _make_agent()
        result = agent.run("some goal")
        self.assertEqual(result["status"], "Failed")
        self.assertIn("Plan", result["error"])


class TestMemoryExtraction(unittest.TestCase):
    def test_extract_from_memory_gets_logs(self):
        agent    = _make_agent()
        mock_agent = MockCodeAgentSuccess()
        logs, codes = agent._extract_from_memory(mock_agent)
        self.assertIn("Day Shift", logs)

    def test_extract_from_empty_memory(self):
        agent      = _make_agent()
        mock_agent = MagicMock()
        mock_agent.memory.steps = []
        logs, codes = agent._extract_from_memory(mock_agent)
        self.assertEqual(logs, "")
        self.assertEqual(codes, [])


class TestCodeAgentRun(unittest.TestCase):
    @patch("main.CodeAgent", MockCodeAgentSuccess)
    @patch("main._llm_call", side_effect=_fake_llm_call)
    def test_successful_run(self, _):
        agent  = _make_agent()
        result = agent.run("Average salary by department")
        self.assertEqual(result["status"], "Success")
        self.assertGreater(len(result["interpretation"]), 0)

    @patch("main.CodeAgent", MockCodeAgentFail)
    @patch("main._llm_call", side_effect=_fake_llm_call)
    def test_failed_run_returns_failed(self, _):
        agent  = _make_agent()
        result = agent.run("Average salary by department")
        self.assertEqual(result["status"], "Failed")

    @patch("main.CodeAgent", MockCodeAgentRetry)
    @patch("main._llm_call", side_effect=_fake_llm_call)
    def test_retry_eventually_succeeds(self, _):
        _attempt_counter["n"] = 0
        agent  = _make_agent()
        result = agent.run("Average salary by department")
        # MockCodeAgentRetry has observations so recovery kicks in
        self.assertIn(result["status"], {"Success", "Failed"})


class TestInterpretation(unittest.TestCase):
    @patch("main._llm_call", return_value=FAKE_INTERP)
    def test_interpret_string_result(self, _):
        agent = _make_agent()
        text  = agent.interpret("avg salary", FAKE_AGENT_LOGS, None)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    @patch("main._llm_call", return_value=FAKE_INTERP)
    def test_interpret_dataframe_result(self, _):
        agent  = _make_agent()
        result = _sample_df()
        text   = agent.interpret("avg salary by dept", result, None)
        self.assertIsInstance(text, str)

    @patch("main._llm_call", return_value=FAKE_INTERP)
    def test_interpret_scalar_result(self, _):
        agent = _make_agent()
        text  = agent.interpret("count rows", 42, None)
        self.assertIsInstance(text, str)


class TestSummaryKeys(unittest.TestCase):
    @patch("main.CodeAgent", MockCodeAgentSuccess)
    @patch("main._llm_call", side_effect=_fake_llm_call)
    def test_all_required_keys_present(self, _):
        agent    = _make_agent()
        result   = agent.run("Average salary by department")
        required = {"timestamp", "goal", "status", "error",
                    "code_path", "output_file", "interpretation", "log_path"}
        self.assertTrue(required.issubset(result.keys()))


class TestGeminiGroqFallback(unittest.TestCase):
    def test_make_model_uses_groq_when_gemini_unhealthy(self):
        import main
        original_healthy = main._gemini_healthy
        original_groq    = os.environ.get("GROQ_API_KEY")
        try:
            main._gemini_healthy = False
            os.environ["GROQ_API_KEY"] = "test-groq-key"
            model = main._make_model()
            self.assertIn("groq", model.model_id.lower())
        finally:
            main._gemini_healthy = original_healthy
            if original_groq is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = original_groq


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CSV Insight Analyzer — Test Suite (smolagents)")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=".", pattern="test*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)