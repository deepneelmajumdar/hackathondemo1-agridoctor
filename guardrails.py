"""
guardrails.py – Agri Doctor
============================
Comprehensive guardrail layer covering all 9 guardrail categories:
  1. Input Guardrails
  2. Output Guardrails
  3. Tool Execution Guardrails
  4. Data Guardrails
  5. Security Guardrails
  6. Operational Guardrails
  7. Human-in-the-Loop Guardrails
  8. Multi-Agent Guardrails
  9. Ethical & Compliance Guardrails
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agri_doctor.guardrails")


# ---------------------------------------------------------------------------
# Shared result dataclass
# ---------------------------------------------------------------------------
@dataclass
class GuardrailResult:
    passed: bool
    category: str
    message: str
    severity: str = "INFO"          # INFO | WARNING | BLOCK
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "category": self.category,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ===========================================================================
# 1. INPUT GUARDRAILS
# ===========================================================================
class InputGuardrails:
    """Validate user inputs before passing them to any AI agent."""

    # Prompt injection patterns
    _INJECTION_PATTERNS: List[str] = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"forget\s+(everything|all)\s+(you|your)",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+(if\s+you\s+are|a)",
        r"disregard\s+(your|all)\s+(previous|prior|earlier)",
        r"jailbreak",
        r"do\s+anything\s+now",
        r"dan\s+mode",
        r"pretend\s+(you\s+are|to\s+be)",
        r"override\s+(your|the)\s+(instructions|rules|guidelines)",
        r"system\s+prompt",
        r"</?(system|user|assistant)>",
    ]

    # Malicious content patterns
    _MALICIOUS_PATTERNS: List[str] = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=\s*['\"]",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"subprocess",
        r"os\.system",
        r"DROP\s+TABLE",
        r"SELECT\s+\*\s+FROM",
        r"INSERT\s+INTO",
        r"DELETE\s+FROM",
        r"UPDATE\s+\w+\s+SET",
        r"UNION\s+SELECT",
        r"OR\s+1\s*=\s*1",
    ]

    # PII patterns
    _PII_PATTERNS: List[Tuple[str, str]] = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        (r"\b\d{16}\b", "credit_card"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
        (r"\b\+?[\d\s\-\(\)]{10,15}\b", "phone_number"),
        (r"\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|boulevard|blvd)\b", "address"),
        (r"\b\d{9}\b", "id_number"),
    ]

    # Inappropriate content keywords
    _INAPPROPRIATE_KEYWORDS: List[str] = [
        "violence", "explicit", "pornography", "abuse", "illegal drugs",
        "weapons", "terrorism", "hate speech", "self-harm",
    ]

    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_TOKENS", "2000")) * 4  # chars ~ tokens

    def validate(self, text: str, context: str = "user_input") -> List[GuardrailResult]:
        results: List[GuardrailResult] = []
        results.extend(self._check_length(text))
        results.extend(self._check_prompt_injection(text))
        results.extend(self._check_malicious_content(text))
        results.extend(self._check_pii(text, context))
        results.extend(self._check_inappropriate_content(text))
        return results

    def _check_length(self, text: str) -> List[GuardrailResult]:
        if len(text) > self.MAX_INPUT_LENGTH:
            return [GuardrailResult(
                passed=False, category="input_validation",
                message=f"Input exceeds maximum length ({len(text)} chars). Max allowed: {self.MAX_INPUT_LENGTH}.",
                severity="BLOCK",
            )]
        return [GuardrailResult(passed=True, category="input_validation", message="Input length OK.")]

    def _check_prompt_injection(self, text: str) -> List[GuardrailResult]:
        lowered = text.lower()
        for pat in self._INJECTION_PATTERNS:
            if re.search(pat, lowered, re.IGNORECASE):
                logger.warning("Prompt injection detected: pattern=%s", pat)
                return [GuardrailResult(
                    passed=False, category="prompt_injection",
                    message="Potential prompt injection detected. Input rejected.",
                    severity="BLOCK", details={"pattern": pat},
                )]
        return [GuardrailResult(passed=True, category="prompt_injection", message="No injection detected.")]

    def _check_malicious_content(self, text: str) -> List[GuardrailResult]:
        for pat in self._MALICIOUS_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                logger.warning("Malicious content detected: pattern=%s", pat)
                return [GuardrailResult(
                    passed=False, category="malicious_input",
                    message="Malicious content detected. Input rejected.",
                    severity="BLOCK", details={"pattern": pat},
                )]
        return [GuardrailResult(passed=True, category="malicious_input", message="No malicious content.")]

    def _check_pii(self, text: str, context: str) -> List[GuardrailResult]:
        found_pii: List[str] = []
        for pat, pii_type in self._PII_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                found_pii.append(pii_type)
        if found_pii:
            logger.warning("PII detected in %s: types=%s", context, found_pii)
            return [GuardrailResult(
                passed=False, category="pii_detection",
                message=f"Possible personal information detected ({', '.join(found_pii)}). Please remove sensitive data.",
                severity="WARNING", details={"pii_types": found_pii},
            )]
        return [GuardrailResult(passed=True, category="pii_detection", message="No PII detected.")]

    def _check_inappropriate_content(self, text: str) -> List[GuardrailResult]:
        lowered = text.lower()
        found = [kw for kw in self._INAPPROPRIATE_KEYWORDS if kw in lowered]
        if found:
            return [GuardrailResult(
                passed=False, category="inappropriate_content",
                message="Input contains inappropriate content. Please keep queries related to crop health.",
                severity="BLOCK", details={"keywords": found},
            )]
        return [GuardrailResult(passed=True, category="inappropriate_content", message="Content appropriate.")]


# ===========================================================================
# 2. OUTPUT GUARDRAILS
# ===========================================================================
class OutputGuardrails:
    """Validate AI-generated responses before displaying to users."""

    _TOXIC_PATTERNS: List[str] = [
        r"\b(kill|murder|destroy|annihilate)\s+(people|humans|farmers)\b",
        r"\bhate\s+(group|crime)\b",
        r"\bterrorist\b",
        r"\bsuicide\b",
    ]

    _SENSITIVE_PATTERNS: List[Tuple[str, str]] = [
        (r"sk-[a-zA-Z0-9]{20,}", "openai_api_key"),
        (r"gsk_[a-zA-Z0-9]{20,}", "groq_api_key"),
        (r"AIza[0-9A-Za-z\-_]{35}", "google_api_key"),
        (r"[a-f0-9]{32}", "potential_secret"),
    ]

    # Hallucination indicators
    _HALLUCINATION_PHRASES: List[str] = [
        "i am certain that",
        "100% guaranteed",
        "this will definitely cure",
        "approved by fda",    # should never make such claims for crops
        "there is no doubt",
        "completely safe with no side effects",
    ]

    MIN_RESPONSE_LENGTH = 50
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_OUTPUT_TOKENS", "1500")) * 5

    def validate(self, text: str, agent_name: str = "agent") -> List[GuardrailResult]:
        results: List[GuardrailResult] = []
        results.extend(self._check_length(text))
        results.extend(self._check_toxicity(text))
        results.extend(self._check_sensitive_info(text))
        results.extend(self._check_hallucination_indicators(text))
        results.extend(self._check_format(text))
        return results

    def _check_length(self, text: str) -> List[GuardrailResult]:
        if len(text) < self.MIN_RESPONSE_LENGTH:
            return [GuardrailResult(
                passed=False, category="response_format",
                message="Response too short – may indicate an incomplete generation.",
                severity="WARNING",
            )]
        if len(text) > self.MAX_RESPONSE_LENGTH:
            return [GuardrailResult(
                passed=False, category="response_format",
                message="Response exceeds maximum length. Output truncated for safety.",
                severity="WARNING",
            )]
        return [GuardrailResult(passed=True, category="response_format", message="Response length OK.")]

    def _check_toxicity(self, text: str) -> List[GuardrailResult]:
        for pat in self._TOXIC_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                logger.warning("Toxic content detected in output: pattern=%s", pat)
                return [GuardrailResult(
                    passed=False, category="toxicity",
                    message="Potentially harmful content detected in AI response. Blocked.",
                    severity="BLOCK", details={"pattern": pat},
                )]
        return [GuardrailResult(passed=True, category="toxicity", message="No toxic content.")]

    def _check_sensitive_info(self, text: str) -> List[GuardrailResult]:
        found: List[str] = []
        for pat, key_type in self._SENSITIVE_PATTERNS:
            if re.search(pat, text):
                found.append(key_type)
        if found:
            return [GuardrailResult(
                passed=False, category="sensitive_info_filter",
                message="Sensitive information (possible API key/secret) detected in output. Redacted.",
                severity="BLOCK", details={"types": found},
            )]
        return [GuardrailResult(passed=True, category="sensitive_info_filter", message="No sensitive info leaked.")]

    def _check_hallucination_indicators(self, text: str) -> List[GuardrailResult]:
        lowered = text.lower()
        found = [p for p in self._HALLUCINATION_PHRASES if p in lowered]
        if found:
            return [GuardrailResult(
                passed=False, category="hallucination_detection",
                message="Response contains overly certain language that may indicate hallucination.",
                severity="WARNING", details={"phrases": found},
            )]
        return [GuardrailResult(passed=True, category="hallucination_detection", message="No hallucination markers.")]

    def _check_format(self, text: str) -> List[GuardrailResult]:
        """Basic format validation – ensure response is readable prose."""
        if text.count("\n") == 0 and len(text) > 500:
            return [GuardrailResult(
                passed=False, category="response_format",
                message="Response appears to be a single unformatted block. Display may be poor.",
                severity="INFO",
            )]
        return [GuardrailResult(passed=True, category="response_format", message="Format OK.")]


# ===========================================================================
# 3. TOOL EXECUTION GUARDRAILS
# ===========================================================================
class ToolExecutionGuardrails:
    """Control how AI agents interact with external tools and APIs."""

    ALLOWED_TOOLS = {"replicate_vision_api", "replicate_text_api", "image_loader"}
    _call_timestamps: List[float] = []
    RATE_LIMIT = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "10"))

    def validate_tool_call(self, tool_name: str, params: Dict[str, Any]) -> GuardrailResult:
        if tool_name not in self.ALLOWED_TOOLS:
            return GuardrailResult(
                passed=False, category="tool_access",
                message=f"Tool '{tool_name}' is not in the approved list.",
                severity="BLOCK",
            )
        if not self._check_rate_limit():
            return GuardrailResult(
                passed=False, category="api_rate_limit",
                message=f"Rate limit exceeded ({self.RATE_LIMIT} calls/min). Please wait.",
                severity="BLOCK",
            )
        if "api_key" in params or "password" in params or "secret" in params:
            return GuardrailResult(
                passed=False, category="credential_exposure",
                message="Credentials must not be passed directly as tool parameters.",
                severity="BLOCK",
            )
        return GuardrailResult(passed=True, category="tool_execution", message=f"Tool '{tool_name}' call approved.")

    def _check_rate_limit(self) -> bool:
        now = time.time()
        self._call_timestamps[:] = [t for t in self._call_timestamps if now - t < 60]
        if len(self._call_timestamps) >= self.RATE_LIMIT:
            return False
        self._call_timestamps.append(now)
        return True

    def validate_api_authorization(self, api_key: Optional[str], service: str) -> GuardrailResult:
        if not api_key or len(api_key.strip()) < 10:
            return GuardrailResult(
                passed=False, category="api_authorization",
                message=f"Invalid or missing API key for {service}.",
                severity="BLOCK",
            )
        # Check key is not a placeholder
        placeholder_patterns = ["your_", "xxx", "placeholder", "changeme", "test123"]
        if any(p in api_key.lower() for p in placeholder_patterns):
            return GuardrailResult(
                passed=False, category="api_authorization",
                message=f"Placeholder API key detected for {service}. Enter a real key.",
                severity="BLOCK",
            )
        return GuardrailResult(passed=True, category="api_authorization", message=f"{service} API key appears valid.")


# ===========================================================================
# 4. DATA GUARDRAILS
# ===========================================================================
class DataGuardrails:
    """Protect data accessed and processed by agents."""

    _ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}
    MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_SIZE_MB", "5")) * 1024 * 1024

    def validate_image(self, image_bytes: bytes, mime_type: str, filename: str) -> List[GuardrailResult]:
        results: List[GuardrailResult] = []
        results.append(self._check_mime_type(mime_type))
        results.append(self._check_file_size(image_bytes))
        results.append(self._check_filename(filename))
        return results

    def _check_mime_type(self, mime_type: str) -> GuardrailResult:
        normalized = mime_type.lower().strip()
        if normalized not in self._ALLOWED_MIME_TYPES:
            return GuardrailResult(
                passed=False, category="data_validation",
                message=f"Unsupported image type '{mime_type}'. Allowed: JPG, JPEG, PNG.",
                severity="BLOCK",
            )
        return GuardrailResult(passed=True, category="data_validation", message="Image MIME type OK.")

    def _check_file_size(self, image_bytes: bytes) -> GuardrailResult:
        size_mb = len(image_bytes) / (1024 * 1024)
        if len(image_bytes) > self.MAX_IMAGE_BYTES:
            return GuardrailResult(
                passed=False, category="data_validation",
                message=f"Image too large ({size_mb:.1f} MB). Max allowed: {self.MAX_IMAGE_BYTES // (1024*1024)} MB.",
                severity="BLOCK",
            )
        return GuardrailResult(passed=True, category="data_validation", message=f"Image size OK ({size_mb:.2f} MB).")

    def _check_filename(self, filename: str) -> GuardrailResult:
        # Prevent path traversal attacks
        if ".." in filename or "/" in filename or "\\" in filename:
            return GuardrailResult(
                passed=False, category="data_privacy",
                message="Invalid filename – path traversal detected.",
                severity="BLOCK",
            )
        return GuardrailResult(passed=True, category="data_privacy", message="Filename OK.")

    def mask_pii(self, text: str) -> str:
        """Anonymize PII in text before logging or storing."""
        for pat, pii_type in InputGuardrails._PII_PATTERNS:
            text = re.sub(pat, f"[REDACTED_{pii_type.upper()}]", text, flags=re.IGNORECASE)
        return text

    def hash_sensitive(self, value: str) -> str:
        """One-way hash for audit logging of sensitive values (e.g. API key fingerprint)."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]


# ===========================================================================
# 5. SECURITY GUARDRAILS
# ===========================================================================
class SecurityGuardrails:
    """Protect the application from security threats."""

    _JAILBREAK_PATTERNS: List[str] = [
        r"developer\s+mode",
        r"unlock\s+your\s+(full\s+)?potential",
        r"no\s+restrictions",
        r"unrestricted\s+mode",
        r"sudo\s+mode",
        r"admin\s+override",
        r"bypass\s+(safety|filter|guardrail)",
        r"without\s+any\s+(restrictions|limitations|filters)",
        r"ignore\s+(ethical|safety)\s+(guidelines|constraints)",
        r"you\s+have\s+no\s+(restrictions|rules|guidelines)",
    ]

    _SQL_PATTERNS: List[str] = [
        r";\s*(drop|delete|truncate|alter|insert|update)\s+",
        r"--\s*$",
        r"/\*.*?\*/",
        r"xp_cmdshell",
        r"EXEC\s*\(",
        r"CAST\s*\(",
    ]

    def check_jailbreak(self, text: str) -> GuardrailResult:
        for pat in self._JAILBREAK_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                logger.warning("Jailbreak attempt detected: pattern=%s", pat)
                return GuardrailResult(
                    passed=False, category="jailbreak_detection",
                    message="Jailbreak attempt detected. Input rejected.",
                    severity="BLOCK", details={"pattern": pat},
                )
        return GuardrailResult(passed=True, category="jailbreak_detection", message="No jailbreak detected.")

    def check_sql_injection(self, text: str) -> GuardrailResult:
        for pat in self._SQL_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return GuardrailResult(
                    passed=False, category="sql_injection",
                    message="SQL injection pattern detected. Input rejected.",
                    severity="BLOCK", details={"pattern": pat},
                )
        return GuardrailResult(passed=True, category="sql_injection", message="No SQL injection detected.")

    def validate_cross_agent_message(self, sender: str, receiver: str, payload: str) -> GuardrailResult:
        """Validate messages passed between agents."""
        allowed_pairs = {("pathologist_agent", "extension_officer_agent")}
        if (sender, receiver) not in allowed_pairs:
            return GuardrailResult(
                passed=False, category="cross_agent_security",
                message=f"Unauthorized agent communication: {sender} → {receiver}.",
                severity="BLOCK",
            )
        if len(payload) < 10:
            return GuardrailResult(
                passed=False, category="cross_agent_security",
                message="Agent handoff payload is too short – possible empty or corrupt message.",
                severity="WARNING",
            )
        return GuardrailResult(passed=True, category="cross_agent_security", message="Cross-agent message validated.")


# ===========================================================================
# 6. OPERATIONAL GUARDRAILS
# ===========================================================================
class OperationalGuardrails:
    """Control agent execution and resource consumption."""

    MAX_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
    MAX_EXEC_TIME = int(os.getenv("MAX_EXECUTION_TIME_SECONDS", "120"))
    BUDGET_LIMIT_USD = float(os.getenv("COST_BUDGET_LIMIT_USD", "5.0"))
    COST_PER_1K_INPUT_TOKENS = 0.0001   # conservative estimate
    COST_PER_1K_OUTPUT_TOKENS = 0.0003

    _iteration_count: int = 0
    _start_time: float = 0.0
    _total_cost: float = 0.0

    def start_session(self) -> None:
        self._iteration_count = 0
        self._start_time = time.time()
        self._total_cost = 0.0

    def check_iteration_limit(self) -> GuardrailResult:
        self._iteration_count += 1
        if self._iteration_count > self.MAX_ITERATIONS:
            return GuardrailResult(
                passed=False, category="max_iterations",
                message=f"Maximum agent iterations ({self.MAX_ITERATIONS}) exceeded. Stopping.",
                severity="BLOCK",
            )
        return GuardrailResult(
            passed=True, category="max_iterations",
            message=f"Iteration {self._iteration_count}/{self.MAX_ITERATIONS} OK.",
        )

    def check_execution_time(self) -> GuardrailResult:
        elapsed = time.time() - self._start_time
        if elapsed > self.MAX_EXEC_TIME:
            return GuardrailResult(
                passed=False, category="execution_time",
                message=f"Execution time limit ({self.MAX_EXEC_TIME}s) exceeded. Session terminated.",
                severity="BLOCK", details={"elapsed_seconds": round(elapsed, 2)},
            )
        return GuardrailResult(
            passed=True, category="execution_time",
            message=f"Elapsed: {elapsed:.1f}s / {self.MAX_EXEC_TIME}s.",
        )

    def track_tokens_and_cost(self, input_tokens: int, output_tokens: int) -> GuardrailResult:
        cost = (input_tokens / 1000 * self.COST_PER_1K_INPUT_TOKENS +
                output_tokens / 1000 * self.COST_PER_1K_OUTPUT_TOKENS)
        self._total_cost += cost
        if self._total_cost > self.BUDGET_LIMIT_USD:
            return GuardrailResult(
                passed=False, category="cost_monitoring",
                message=f"Budget limit (${self.BUDGET_LIMIT_USD}) exceeded. Cost so far: ${self._total_cost:.4f}.",
                severity="BLOCK", details={"total_cost_usd": round(self._total_cost, 4)},
            )
        return GuardrailResult(
            passed=True, category="cost_monitoring",
            message=f"Cost this session: ${self._total_cost:.4f} / ${self.BUDGET_LIMIT_USD}.",
            details={"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": round(cost, 6)},
        )


# ===========================================================================
# 7. HUMAN-IN-THE-LOOP GUARDRAILS
# ===========================================================================
class HumanInTheLoopGuardrails:
    """Ensure human review for critical or high-risk outputs."""

    # Conditions that require human escalation
    _ESCALATION_TRIGGERS: List[str] = [
        "emergency", "severe infection", "total crop failure", "quarantine",
        "outbreak", "notifiable disease", "quarantine pest",
        "unable to identify", "consult expert", "laboratory confirmation",
    ]

    def check_needs_human_review(self, diagnosis_text: str) -> GuardrailResult:
        lowered = diagnosis_text.lower()
        triggers_found = [t for t in self._ESCALATION_TRIGGERS if t in lowered]
        if triggers_found:
            return GuardrailResult(
                passed=True,  # output is still shown
                category="human_escalation",
                message=(
                    "⚠️ This diagnosis contains critical findings. "
                    "We recommend escalating to a certified agricultural extension officer or plant pathologist."
                ),
                severity="WARNING",
                details={"triggers": triggers_found},
            )
        return GuardrailResult(
            passed=True, category="human_escalation",
            message="No immediate escalation required.",
        )

    def add_disclaimer(self, text: str) -> str:
        """Prepend mandatory AI disclaimer to any advisory output."""
        disclaimer = (
            "⚠️ DISCLAIMER: This is an AI-generated preliminary assessment only. "
            "It is NOT a confirmed laboratory diagnosis. Always consult a certified "
            "agricultural extension officer or plant pathologist before taking action. "
            "Follow all local agricultural regulations and pesticide safety guidelines.\n\n"
        )
        return disclaimer + text


# ===========================================================================
# 8. MULTI-AGENT GUARDRAILS
# ===========================================================================
class MultiAgentGuardrails:
    """Ensure safe coordination between Pathologist and Extension Officer agents."""

    DEFINED_AGENTS = {"pathologist_agent", "extension_officer_agent"}
    _execution_log: List[Dict[str, Any]] = []
    _loop_detection_window: int = 3

    def validate_agent_registration(self, agent_name: str) -> GuardrailResult:
        if agent_name not in self.DEFINED_AGENTS:
            return GuardrailResult(
                passed=False, category="agent_access_control",
                message=f"Unknown agent '{agent_name}'. Only defined agents may execute.",
                severity="BLOCK",
            )
        return GuardrailResult(passed=True, category="agent_access_control", message=f"Agent '{agent_name}' registered.")

    def validate_handoff(self, from_agent: str, to_agent: str, output: str) -> GuardrailResult:
        """Validate the output handed from Pathologist to Extension Officer."""
        if not output or len(output.strip()) < 30:
            return GuardrailResult(
                passed=False, category="handoff_accuracy",
                message=f"Handoff from {from_agent} to {to_agent} failed: output too short or empty.",
                severity="BLOCK",
            )
        required_keywords = ["disease", "symptom", "diagnosis", "crop", "plant", "leaf", "infection"]
        lowered = output.lower()
        if not any(kw in lowered for kw in required_keywords):
            return GuardrailResult(
                passed=False, category="handoff_accuracy",
                message="Pathologist output does not appear to contain crop diagnosis information.",
                severity="WARNING",
            )
        return GuardrailResult(
            passed=True, category="handoff_accuracy",
            message=f"Handoff from {from_agent} to {to_agent} validated successfully.",
        )

    def detect_loop(self, agent_name: str) -> GuardrailResult:
        self._execution_log.append({"agent": agent_name, "time": time.time()})
        recent = self._execution_log[-self._loop_detection_window:]
        if (len(recent) == self._loop_detection_window and
                all(e["agent"] == agent_name for e in recent)):
            return GuardrailResult(
                passed=False, category="loop_detection",
                message=f"Infinite loop detected: agent '{agent_name}' called {self._loop_detection_window} times consecutively.",
                severity="BLOCK",
            )
        return GuardrailResult(passed=True, category="loop_detection", message="No loop detected.")

    def verify_cross_agent_output(self, agent_name: str, output: str) -> GuardrailResult:
        """Lightweight consistency check on cross-agent outputs."""
        if len(output) < 50:
            return GuardrailResult(
                passed=False, category="cross_agent_verification",
                message=f"Agent '{agent_name}' produced an insufficient output.",
                severity="WARNING",
            )
        return GuardrailResult(
            passed=True, category="cross_agent_verification",
            message=f"Output from '{agent_name}' passed cross-agent verification.",
        )


# ===========================================================================
# 9. ETHICAL & COMPLIANCE GUARDRAILS
# ===========================================================================
class EthicalComplianceGuardrails:
    """Ensure responsible and compliant AI behaviour."""

    _BIAS_INDICATORS: List[str] = [
        r"\b(only|always|never)\s+(farmers\s+in|people\s+from|crops\s+in)\s+\w+\b",
        r"\b(small|poor|uneducated)\s+farmers?\b",
        r"\b(third\s+world|primitive)\s+(farming|agriculture)\b",
    ]

    def check_bias(self, text: str) -> GuardrailResult:
        for pat in self._BIAS_INDICATORS:
            if re.search(pat, text, re.IGNORECASE):
                return GuardrailResult(
                    passed=False, category="bias_detection",
                    message="Potential bias detected in AI output. Review before displaying.",
                    severity="WARNING", details={"pattern": pat},
                )
        return GuardrailResult(passed=True, category="bias_detection", message="No bias indicators found.")

    def add_audit_entry(self, audit_log: List[Dict], event: str, details: Dict) -> None:
        """Append an audit trail entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "details": details,
        }
        audit_log.append(entry)
        if os.getenv("ENABLE_AUDIT_TRAIL", "true").lower() == "true":
            logger.info("AUDIT | %s | %s", event, details)

    def check_regulatory_compliance(self, text: str) -> GuardrailResult:
        """Flag responses that make unverified regulatory claims."""
        non_compliant_phrases = [
            "guaranteed to eliminate", "100% effective", "no side effects",
            "approved by ministry of agriculture", "government certified",
        ]
        lowered = text.lower()
        found = [p for p in non_compliant_phrases if p in lowered]
        if found:
            return GuardrailResult(
                passed=False, category="regulatory_compliance",
                message="Response contains unverified regulatory claims. Please review.",
                severity="WARNING", details={"phrases": found},
            )
        return GuardrailResult(passed=True, category="regulatory_compliance", message="No compliance issues found.")


# ===========================================================================
# GUARDRAIL ORCHESTRATOR – single entry point
# ===========================================================================
class GuardrailOrchestrator:
    """
    Unified orchestrator that runs all relevant guardrails and
    returns a consolidated pass/fail decision.
    """

    def __init__(self) -> None:
        self.input_g = InputGuardrails()
        self.output_g = OutputGuardrails()
        self.tool_g = ToolExecutionGuardrails()
        self.data_g = DataGuardrails()
        self.security_g = SecurityGuardrails()
        self.ops_g = OperationalGuardrails()
        self.hitl_g = HumanInTheLoopGuardrails()
        self.multi_g = MultiAgentGuardrails()
        self.ethics_g = EthicalComplianceGuardrails()
        self.audit_log: List[Dict] = []

    def validate_user_input(self, text: str) -> Tuple[bool, List[GuardrailResult]]:
        results = (
            self.input_g.validate(text)
            + [self.security_g.check_jailbreak(text)]
            + [self.security_g.check_sql_injection(text)]
        )
        blocked = any(r.severity == "BLOCK" and not r.passed for r in results)
        self.ethics_g.add_audit_entry(
            self.audit_log, "user_input_validated",
            {"passed": not blocked, "result_count": len(results)},
        )
        return not blocked, results

    def validate_agent_output(self, text: str, agent_name: str) -> Tuple[bool, List[GuardrailResult]]:
        results = (
            self.output_g.validate(text, agent_name)
            + [self.ethics_g.check_bias(text)]
            + [self.ethics_g.check_regulatory_compliance(text)]
            + [self.hitl_g.check_needs_human_review(text)]
        )
        blocked = any(r.severity == "BLOCK" and not r.passed for r in results)
        self.ethics_g.add_audit_entry(
            self.audit_log, "agent_output_validated",
            {"agent": agent_name, "passed": not blocked},
        )
        return not blocked, results

    def validate_image_upload(self, image_bytes: bytes, mime_type: str, filename: str) -> Tuple[bool, List[GuardrailResult]]:
        results = self.data_g.validate_image(image_bytes, mime_type, filename)
        blocked = any(r.severity == "BLOCK" and not r.passed for r in results)
        self.ethics_g.add_audit_entry(
            self.audit_log, "image_upload_validated",
            {"filename": filename, "size_bytes": len(image_bytes), "passed": not blocked},
        )
        return not blocked, results

    def validate_agent_handoff(self, from_agent: str, to_agent: str, payload: str) -> Tuple[bool, List[GuardrailResult]]:
        results = [
            self.security_g.validate_cross_agent_message(from_agent, to_agent, payload),
            self.multi_g.validate_handoff(from_agent, to_agent, payload),
            self.multi_g.detect_loop(to_agent),
        ]
        blocked = any(r.severity == "BLOCK" and not r.passed for r in results)
        self.ethics_g.add_audit_entry(
            self.audit_log, "agent_handoff_validated",
            {"from": from_agent, "to": to_agent, "passed": not blocked},
        )
        return not blocked, results

    def get_blocking_messages(self, results: List[GuardrailResult]) -> List[str]:
        return [r.message for r in results if not r.passed and r.severity == "BLOCK"]

    def get_warning_messages(self, results: List[GuardrailResult]) -> List[str]:
        return [r.message for r in results if not r.passed and r.severity == "WARNING"]
