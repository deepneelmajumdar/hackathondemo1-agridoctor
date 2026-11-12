"""
dashboard.py – Agri Doctor
===========================
LLM Agent Performance Dashboard and Evaluation Metrics Engine.

Covers all 6 evaluation categories:
  1. Accuracy and Quality Metrics
  2. RAG Evaluation Metrics
  3. Agentic AI Evaluation Metrics
  4. Safety and Guardrail Metrics
  5. Performance and Cost Metrics
  6. Multi-Agent System Evaluation
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentRun:
    """Stores all telemetry for a single agent invocation."""
    agent_name: str
    start_time: float
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    input_text: str = ""
    output_text: str = ""
    success: bool = False
    error_message: str = ""
    guardrail_blocks: int = 0
    guardrail_warnings: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    iteration_count: int = 0

    @property
    def latency_ms(self) -> float:
        return round((self.end_time - self.start_time) * 1000, 2)

    @property
    def cost_usd(self) -> float:
        inp = self.input_tokens / 1000 * 0.0001
        out = self.output_tokens / 1000 * 0.0003
        return round(inp + out, 6)

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.end_time - self.start_time
        if elapsed <= 0:
            return 0.0
        return round(self.output_tokens / elapsed, 2)


@dataclass
class SessionMetrics:
    """Aggregated metrics for a full Agri Doctor session (both agents)."""
    session_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    agent_runs: List[AgentRun] = field(default_factory=list)
    guardrail_results: List[Dict[str, Any]] = field(default_factory=list)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    # Derived quality scores (set after evaluation)
    hallucination_rate: float = 0.0
    toxicity_score: float = 0.0
    bias_score: float = 0.0
    task_completion_rate: float = 0.0
    handoff_accuracy: float = 0.0
    guardrail_intervention_rate: float = 0.0

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.agent_runs)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.agent_runs)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.agent_runs), 6)

    @property
    def total_latency_ms(self) -> float:
        if not self.agent_runs:
            return 0.0
        start = min(r.start_time for r in self.agent_runs)
        end = max(r.end_time for r in self.agent_runs if r.end_time > 0)
        return round((end - start) * 1000, 2)

    @property
    def error_rate(self) -> float:
        if not self.agent_runs:
            return 0.0
        failed = sum(1 for r in self.agent_runs if not r.success)
        return round(failed / len(self.agent_runs), 4)


# ---------------------------------------------------------------------------
# Metric calculation helpers
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """Pure-function metric calculations – no side effects."""

    # ---- 1. Accuracy & Quality ----

    @staticmethod
    def ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    @staticmethod
    def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> float:
        """Simplified corpus-level BLEU (no brevity penalty for single sentence)."""
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        if not hyp_tokens or not ref_tokens:
            return 0.0
        scores = []
        for n in range(1, max_n + 1):
            hyp_ng = MetricsCalculator.ngrams(hyp_tokens, n)
            ref_ng = MetricsCalculator.ngrams(ref_tokens, n)
            clipped = sum(min(hyp_ng[g], ref_ng[g]) for g in hyp_ng)
            total = sum(hyp_ng.values())
            if total == 0:
                scores.append(0.0)
            else:
                scores.append(clipped / total)
        # Geometric mean
        if all(s == 0 for s in scores):
            return 0.0
        log_avg = sum(math.log(s + 1e-10) for s in scores) / len(scores)
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
        return round(bp * math.exp(log_avg), 4)

    @staticmethod
    def rouge_l(hypothesis: str, reference: str) -> float:
        """ROUGE-L based on LCS length."""
        hyp = hypothesis.lower().split()
        ref = reference.lower().split()
        if not hyp or not ref:
            return 0.0
        m, n = len(hyp), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if hyp[i-1] == ref[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        precision = lcs / m if m else 0
        recall = lcs / n if n else 0
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 4)

    @staticmethod
    def semantic_similarity_approx(text_a: str, text_b: str) -> float:
        """Approximation via Jaccard similarity on word sets."""
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a or not set_b:
            return 0.0
        inter = set_a & set_b
        union = set_a | set_b
        return round(len(inter) / len(union), 4)

    @staticmethod
    def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    # ---- 2. RAG Evaluation ----

    @staticmethod
    def faithfulness_score(response: str, context: str) -> float:
        """Approx: fraction of response sentences supported by context keywords."""
        sentences = [s.strip() for s in re.split(r"[.!?]", response) if s.strip()]
        if not sentences:
            return 0.0
        context_words = set(context.lower().split())
        supported = 0
        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = sent_words & context_words
            if len(overlap) / max(len(sent_words), 1) > 0.2:
                supported += 1
        return round(supported / len(sentences), 4)

    @staticmethod
    def answer_relevancy(response: str, query: str) -> float:
        """Jaccard similarity between response and query."""
        return MetricsCalculator.semantic_similarity_approx(response, query)

    # ---- 3. Safety Metrics ----

    _HALLUCINATION_MARKERS = [
        "i am certain", "definitely", "100% guaranteed", "will definitely",
        "without any doubt", "absolutely certain",
    ]
    _TOXIC_PATTERNS = [r"\b(kill|murder|destroy)\s+(people|humans)\b", r"\bterrorist\b"]

    @staticmethod
    def hallucination_rate(text: str) -> float:
        """Returns a 0–1 score; higher = more likely hallucination markers present."""
        lowered = text.lower()
        hits = sum(1 for m in MetricsCalculator._HALLUCINATION_MARKERS if m in lowered)
        return round(min(hits / max(len(text.split()) / 50, 1), 1.0), 4)

    @staticmethod
    def toxicity_score(text: str) -> float:
        hits = sum(1 for p in MetricsCalculator._TOXIC_PATTERNS
                   if re.search(p, text, re.IGNORECASE))
        return round(min(hits * 0.25, 1.0), 4)

    @staticmethod
    def bias_score(text: str) -> float:
        bias_patterns = [
            r"\b(only|always|never)\s+(farmers\s+in|crops\s+in)\s+\w+\b",
            r"\b(poor|uneducated)\s+farmers?\b",
        ]
        hits = sum(1 for p in bias_patterns if re.search(p, text, re.IGNORECASE))
        return round(min(hits * 0.3, 1.0), 4)

    @staticmethod
    def pii_leakage_rate(text: str) -> float:
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        ]
        hits = sum(1 for p in pii_patterns if re.search(p, text, re.IGNORECASE))
        return round(min(hits * 0.5, 1.0), 4)


# ---------------------------------------------------------------------------
# Session metric evaluator
# ---------------------------------------------------------------------------

class SessionEvaluator:
    """Evaluate a completed SessionMetrics object and annotate derived scores."""

    def evaluate(self, session: SessionMetrics) -> SessionMetrics:
        """Run all evaluations and populate derived fields."""
        if not session.agent_runs:
            return session

        # Task completion
        completed = sum(1 for r in session.agent_runs if r.success)
        session.task_completion_rate = round(completed / len(session.agent_runs), 4)

        # Handoff accuracy: check if Extension Officer received pathologist output
        if len(session.agent_runs) >= 2:
            pathologist_run = next((r for r in session.agent_runs
                                    if r.agent_name == "pathologist_agent"), None)
            ext_run = next((r for r in session.agent_runs
                            if r.agent_name == "extension_officer_agent"), None)
            if pathologist_run and ext_run and pathologist_run.output_text:
                # Handoff accuracy: did ext officer use pathologist's key terms?
                session.handoff_accuracy = MetricsCalculator.semantic_similarity_approx(
                    ext_run.input_text, pathologist_run.output_text
                )
            else:
                session.handoff_accuracy = 0.0

        # Aggregate safety metrics from outputs
        all_output = " ".join(r.output_text for r in session.agent_runs)
        session.hallucination_rate = MetricsCalculator.hallucination_rate(all_output)
        session.toxicity_score = MetricsCalculator.toxicity_score(all_output)
        session.bias_score = MetricsCalculator.bias_score(all_output)

        # Guardrail intervention rate
        total_checks = len(session.guardrail_results)
        interventions = sum(1 for r in session.guardrail_results
                            if not r.get("passed", True))
        session.guardrail_intervention_rate = (
            round(interventions / total_checks, 4) if total_checks > 0 else 0.0
        )

        return session


# ---------------------------------------------------------------------------
# Streamlit Dashboard renderer
# ---------------------------------------------------------------------------

def render_dashboard(session: SessionMetrics) -> None:
    """Render the full LLM performance dashboard inside Streamlit."""

    evaluator = SessionEvaluator()
    session = evaluator.evaluate(session)

    st.markdown("---")
    st.header("📊 LLM Agent Performance Dashboard")
    st.caption(f"Session ID: `{session.session_id}` | {session.timestamp}")

    # ── Top KPI row ──────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Task Completion", f"{session.task_completion_rate * 100:.1f}%")
    col2.metric("Total Tokens", f"{session.total_input_tokens + session.total_output_tokens:,}")
    col3.metric("Total Cost (USD)", f"${session.total_cost_usd:.4f}")
    col4.metric("End-to-End Latency", f"{session.total_latency_ms:.0f} ms")
    col5.metric("Error Rate", f"{session.error_rate * 100:.1f}%")

    st.markdown("---")

    # ── Tabs for each category ───────────────────────────────────────────────
    tabs = st.tabs([
        "🎯 Accuracy & Quality",
        "📚 RAG Evaluation",
        "🤖 Agentic Metrics",
        "🛡️ Safety & Guardrails",
        "⚡ Performance & Cost",
        "🔗 Multi-Agent System",
        "📋 Audit Trail",
    ])

    # ── 1. Accuracy & Quality ─────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Accuracy and Quality Metrics")
        if len(session.agent_runs) >= 2:
            path_out = next((r.output_text for r in session.agent_runs
                             if r.agent_name == "pathologist_agent"), "")
            ext_out = next((r.output_text for r in session.agent_runs
                            if r.agent_name == "extension_officer_agent"), "")

            bleu = MetricsCalculator.bleu_score(ext_out, path_out)
            rouge = MetricsCalculator.rouge_l(ext_out, path_out)
            sem_sim = MetricsCalculator.semantic_similarity_approx(ext_out, path_out)

            qa_col1, qa_col2, qa_col3 = st.columns(3)
            qa_col1.metric("BLEU Score", f"{bleu:.4f}", help="N-gram precision (Extension Officer vs. Pathologist output)")
            qa_col2.metric("ROUGE-L", f"{rouge:.4f}", help="LCS-based recall")
            qa_col3.metric("Semantic Similarity", f"{sem_sim:.4f}", help="Jaccard word-set overlap")
        else:
            st.info("Accuracy metrics require both agents to have run.")

        st.markdown("**Metric Definitions**")
        metrics_table = {
            "Metric": ["BLEU", "ROUGE-L", "Semantic Similarity", "Faithfulness", "Answer Relevancy"],
            "Description": [
                "N-gram precision of generated text vs. reference",
                "Longest common subsequence F1 score",
                "Jaccard similarity between word sets",
                "Fraction of output supported by retrieved context",
                "Relevance of response to the original query",
            ],
            "Ideal": ["High (↑)", "High (↑)", "High (↑)", "High (↑)", "High (↑)"],
        }
        _render_table(metrics_table)

    # ── 2. RAG Evaluation ────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("RAG Evaluation Metrics")
        if session.agent_runs:
            all_output = " ".join(r.output_text for r in session.agent_runs)
            all_input = " ".join(r.input_text for r in session.agent_runs)
            faithfulness = MetricsCalculator.faithfulness_score(all_output, all_input)
            relevancy = MetricsCalculator.answer_relevancy(all_output, all_input)
            groundedness = MetricsCalculator.semantic_similarity_approx(all_output, all_input)

            r1, r2, r3 = st.columns(3)
            r1.metric("Faithfulness", f"{faithfulness:.4f}")
            r2.metric("Answer Relevancy", f"{relevancy:.4f}")
            r3.metric("Groundedness", f"{groundedness:.4f}")
        else:
            st.info("No agent runs recorded yet.")

        st.markdown("**Metric Definitions**")
        rag_table = {
            "Metric": ["Context Precision", "Context Recall", "Faithfulness",
                       "Answer Relevancy", "Groundedness"],
            "Description": [
                "Relevant documents in top-k retrieval",
                "All relevant documents retrieved",
                "Answer supported by retrieved context",
                "Answer relevant to user query",
                "Response grounded in source data",
            ],
            "Status": ["Simulated", "Simulated", "Calculated", "Calculated", "Calculated"],
        }
        _render_table(rag_table)

    # ── 3. Agentic AI Metrics ─────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Agentic AI Evaluation Metrics")

        a1, a2, a3 = st.columns(3)
        a1.metric("Task Completion Rate", f"{session.task_completion_rate * 100:.1f}%")
        a2.metric("Handoff Accuracy", f"{session.handoff_accuracy:.4f}")
        a3.metric("Guardrail Intervention Rate", f"{session.guardrail_intervention_rate * 100:.1f}%")

        st.markdown("**Per-Agent Performance**")
        for run in session.agent_runs:
            with st.expander(f"🤖 {run.agent_name.replace('_', ' ').title()}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status", "✅ Success" if run.success else "❌ Failed")
                c2.metric("Latency", f"{run.latency_ms:.0f} ms")
                c3.metric("Tokens/sec", f"{run.tokens_per_second:.1f}")
                c4.metric("Tool Calls", run.tool_calls)
                if run.error_message:
                    st.error(f"Error: {run.error_message}")

        st.markdown("**Metric Definitions**")
        agent_table = {
            "Metric": ["Task Completion Rate", "Goal Achievement", "Tool Call Accuracy",
                       "Handoff Accuracy", "Loop Detection", "Agent Coordination"],
            "Description": [
                "% of tasks successfully completed",
                "Agent achieved its assigned objective",
                "Correct tools selected and invoked",
                "Tasks correctly transferred between agents",
                "Repetitive agent cycles detected",
                "Effective inter-agent collaboration",
            ],
        }
        _render_table(agent_table)

    # ── 4. Safety & Guardrails ────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Safety and Guardrail Evaluation Metrics")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Hallucination Rate",
                  f"{session.hallucination_rate * 100:.2f}%",
                  delta="Low is better",
                  delta_color="inverse")
        s2.metric("Toxicity Score",
                  f"{session.toxicity_score:.4f}",
                  delta="Low is better",
                  delta_color="inverse")
        s3.metric("Bias Score",
                  f"{session.bias_score:.4f}",
                  delta="Low is better",
                  delta_color="inverse")
        s4.metric("Guardrail Intervention Rate",
                  f"{session.guardrail_intervention_rate * 100:.1f}%")

        st.markdown("**Guardrail Check Summary**")
        if session.guardrail_results:
            passed = sum(1 for r in session.guardrail_results if r.get("passed", True))
            failed = len(session.guardrail_results) - passed
            g1, g2 = st.columns(2)
            g1.metric("Checks Passed", passed, delta="✅")
            g2.metric("Checks Failed/Warned", failed,
                      delta="⚠️" if failed > 0 else "✅",
                      delta_color="inverse" if failed > 0 else "normal")

            with st.expander("View All Guardrail Results"):
                for r in session.guardrail_results:
                    icon = "✅" if r.get("passed", True) else ("🚫" if r.get("severity") == "BLOCK" else "⚠️")
                    st.markdown(f"{icon} **{r.get('category', 'unknown')}** – {r.get('message', '')}")
        else:
            st.info("No guardrail results recorded.")

        st.markdown("**Metric Definitions**")
        safety_table = {
            "Metric": ["Hallucination Rate", "Toxicity Score", "Bias Score",
                       "Prompt Injection Resistance", "PII Leakage Rate",
                       "Policy Violation Rate", "Guardrail Intervention Rate"],
            "Description": [
                "% of responses with unsupported claims",
                "Harmful/offensive content measure",
                "Potential bias in generated responses",
                "Resistance to malicious instructions",
                "Accidental personal info exposure",
                "Responses violating defined policies",
                "% of outputs blocked or modified by guardrails",
            ],
        }
        _render_table(safety_table)

    # ── 5. Performance & Cost ─────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Performance and Cost Metrics")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("End-to-End Latency", f"{session.total_latency_ms:.0f} ms")
        p2.metric("Input Tokens", f"{session.total_input_tokens:,}")
        p3.metric("Output Tokens", f"{session.total_output_tokens:,}")
        p4.metric("Total Cost (USD)", f"${session.total_cost_usd:.6f}")

        st.markdown("**Per-Agent Token & Cost Breakdown**")
        for run in session.agent_runs:
            with st.expander(f"💰 {run.agent_name.replace('_', ' ').title()} – Tokens & Cost"):
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Input Tokens", f"{run.input_tokens:,}")
                tc2.metric("Output Tokens", f"{run.output_tokens:,}")
                tc3.metric("Latency", f"{run.latency_ms:.0f} ms")
                tc4.metric("Cost", f"${run.cost_usd:.6f}")

        st.markdown("**Metric Definitions**")
        perf_table = {
            "Metric": ["Response Latency", "TTFT", "Tokens/sec",
                       "Input Tokens", "Output Tokens",
                       "Cost per Request", "Error Rate", "Throughput"],
            "Description": [
                "Time to generate full response",
                "Time to first response token",
                "Token generation speed",
                "Tokens sent to LLM",
                "Tokens generated by LLM",
                "Cost per LLM API call",
                "% of failed requests",
                "Requests processed per second",
            ],
        }
        _render_table(perf_table)

    # ── 6. Multi-Agent System ─────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Multi-Agent System Evaluation")

        ma1, ma2, ma3 = st.columns(3)
        ma1.metric("Total Agents", len(session.agent_runs))
        ma2.metric("Workflow Completion",
                   "✅ Complete" if session.task_completion_rate == 1.0 else "⚠️ Partial")
        ma3.metric("Inter-Agent Consistency",
                   f"{session.handoff_accuracy:.4f}")

        st.markdown("**Agent Execution Timeline**")
        if session.agent_runs:
            base_time = min(r.start_time for r in session.agent_runs)
            for run in session.agent_runs:
                rel_start = (run.start_time - base_time) * 1000
                rel_end = (run.end_time - base_time) * 1000
                bar_width = max(int(rel_end - rel_start), 10)
                status_icon = "✅" if run.success else "❌"
                st.markdown(
                    f"**{status_icon} {run.agent_name.replace('_', ' ').title()}**  "
                    f"  Start: {rel_start:.0f}ms → End: {rel_end:.0f}ms  "
                    f"  (Duration: {run.latency_ms:.0f}ms)"
                )
                st.progress(min(bar_width / 5000, 1.0))

        st.markdown("**Metric Definitions**")
        ma_table = {
            "Metric": ["Total Execution Time", "End-to-End Completion",
                       "Supervisor Accuracy", "Conflict Resolution",
                       "Inter-Agent Consistency", "Task Distribution",
                       "Agent Communication Success"],
            "Description": [
                "Wall-clock time for full workflow",
                "All agents completed successfully",
                "Accuracy of orchestrating agent",
                "Conflicts between agents resolved",
                "Consistency across agent outputs",
                "Efficient task assignment",
                "Message passing without errors",
            ],
        }
        _render_table(ma_table)

    # ── 7. Audit Trail ────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("📋 Audit Trail")
        if session.audit_log:
            for entry in reversed(session.audit_log[-50:]):  # Last 50 entries
                icon = "🔵"
                st.markdown(
                    f"{icon} `{entry.get('timestamp', '')}` | "
                    f"**{entry.get('event', 'unknown')}** | "
                    f"{entry.get('details', {})}"
                )
        else:
            st.info("No audit entries recorded for this session.")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _render_table(data: Dict[str, List]) -> None:
    """Render a plain dict-of-lists as an HTML table in Streamlit."""
    cols = list(data.keys())
    rows = list(zip(*[data[c] for c in cols]))
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    st.markdown(f"{header}\n{separator}\n{body}")


def create_session(session_id: Optional[str] = None) -> SessionMetrics:
    """Factory to create a new SessionMetrics object."""
    sid = session_id or f"session_{int(time.time())}"
    return SessionMetrics(session_id=sid)


def start_agent_run(session: SessionMetrics, agent_name: str) -> AgentRun:
    """Create and register a new AgentRun."""
    run = AgentRun(agent_name=agent_name, start_time=time.time())
    session.agent_runs.append(run)
    return run


def finish_agent_run(run: AgentRun, success: bool,
                     output_text: str = "",
                     input_tokens: int = 0,
                     output_tokens: int = 0,
                     error: str = "") -> None:
    """Mark an AgentRun as complete."""
    run.end_time = time.time()
    run.success = success
    run.output_text = output_text
    run.input_tokens = input_tokens
    run.output_tokens = output_tokens
    run.error_message = error
    run.tool_calls += 1
    if success:
        run.tool_successes += 1
