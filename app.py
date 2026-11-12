"""
app.py – Agri Doctor
=====================
A two-agent Streamlit application that diagnoses crop diseases from leaf/plant
images and generates farmer-friendly remedial advisory reports.

Agents:
  1. Pathologist Agent        – IBM Granite Vision via Replicate (image analysis)
  2. Extension Officer Agent  – IBM Granite 4 via Replicate (text advisory)

Both agents access IBM Granite models through the Replicate API.
All inputs and outputs pass through the comprehensive GuardrailOrchestrator.
A full LLM performance dashboard is rendered after each diagnosis.
"""

from __future__ import annotations

import base64
import io
import os
import time
import uuid
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from dashboard import (
    SessionMetrics,
    create_session,
    finish_agent_run,
    render_dashboard,
    start_agent_run,
)
from guardrails import GuardrailOrchestrator

# ---------------------------------------------------------------------------
# Load environment variables from .env (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agri Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants / configurable model names
# ---------------------------------------------------------------------------
DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "ibm-granite/granite-3-2-vision-instruct")
DEFAULT_TEXT_MODEL = os.getenv("TEXT_MODEL", "ibm-granite/granite-4-0-tiny-preview")
REPLICATE_API_URL = os.getenv("REPLICATE_API_URL", "https://api.replicate.com/v1/models")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ---------------------------------------------------------------------------
# Guardrail orchestrator (session-scoped singleton)
# ---------------------------------------------------------------------------
if "guardrails" not in st.session_state:
    st.session_state.guardrails = GuardrailOrchestrator()

if "session_metrics" not in st.session_state:
    st.session_state.session_metrics = None

if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None

if "advisory_result" not in st.session_state:
    st.session_state.advisory_result = None


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant-under-sun.png", width=80)
    st.title("🌿 Agri Doctor")
    st.caption("AI-powered crop disease diagnosis and advisory")

    st.markdown("---")
    st.subheader("🔑 API Configuration")

    replicate_api_token = st.text_input(
        "Replicate API Token",
        type="password",
        value=os.getenv("REPLICATE_API_TOKEN", ""),
        placeholder="r8_...",
        help="Required for both agents (IBM Granite models via Replicate). Never stored.",
    )

    st.markdown("---")
    st.subheader("⚙️ Model Configuration")
    vision_model = st.text_input(
        "Vision Model (Pathologist Agent)",
        value=DEFAULT_VISION_MODEL,
        help="IBM Granite Vision model identifier on Replicate.",
    )
    text_model = st.text_input(
        "Text Model (Extension Officer Agent)",
        value=DEFAULT_TEXT_MODEL,
        help="IBM Granite 4 model identifier on Replicate.",
    )

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown(
        "Agri Doctor uses two specialized AI agents:\n"
        "- 🔬 **Pathologist Agent** – IBM Granite Vision (Replicate)\n"
        "- 👨‍🌾 **Extension Officer Agent** – IBM Granite 4 (Replicate)\n\n"
        "All outputs are AI-generated preliminary assessments. "
        "Consult a certified agricultural expert before taking action."
    )


# ===========================================================================
# MAIN PAGE
# ===========================================================================
st.title("🌿 Agri Doctor – AI Crop Disease Diagnosis")
st.markdown(
    "> **AI-Powered Agricultural Advisory System** | "
    "Upload a crop/leaf image to receive a disease diagnosis and actionable remedial advice."
)

# ── Disclaimer banner ────────────────────────────────────────────────────────
st.warning(
    "⚠️ **IMPORTANT DISCLAIMER**: All diagnoses are AI-generated preliminary assessments only. "
    "They are NOT confirmed laboratory diagnoses. Always consult a certified agricultural "
    "extension officer or plant pathologist before applying any treatment. "
    "Follow all local agricultural regulations and pesticide safety guidelines."
)

st.markdown("---")

# ===========================================================================
# IMAGE UPLOAD
# ===========================================================================
col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.subheader("📤 Upload Crop Image")
    uploaded_file = st.file_uploader(
        "Upload a clear image of the affected crop leaf or plant",
        type=list(ALLOWED_EXTENSIONS),
        help="Accepted formats: JPG, JPEG, PNG. Max size: 5 MB.",
    )

    additional_context = st.text_area(
        "Additional Context (optional)",
        placeholder="E.g., crop type, region, recent weather, symptoms duration...",
        max_chars=500,
        help="Provide extra context to improve diagnosis accuracy.",
    )

with col_preview:
    if uploaded_file is not None:
        st.subheader("🖼️ Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        st.caption(
            f"File: `{uploaded_file.name}` | "
            f"Size: {len(uploaded_file.getvalue()) / 1024:.1f} KB | "
            f"Format: {uploaded_file.type}"
        )

st.markdown("---")

# ===========================================================================
# DIAGNOSE BUTTON
# ===========================================================================
diagnose_btn = st.button(
    "🔬 Diagnose Crop",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def call_replicate_vision(
    api_token: str,
    image_b64: str,
    mime_type: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, int, int]:
    """
    Call IBM Granite Vision on Replicate using the predictions endpoint.
    Replicate model format: owner/model-name  (e.g. ibm-granite/granite-3-2-vision-instruct)
    Returns (response_text, input_tokens, output_tokens).
    Never logs or exposes the API token.

    Replicate runs the model synchronously when `prefer: wait` header is set,
    returning the output in a single response (no polling required for fast models).
    """
    # Build the data-URI for the image so Replicate receives it inline
    image_data_uri = f"data:{mime_type};base64,{image_b64}"

    url = f"{REPLICATE_API_URL}/{model}/predictions"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait",          # synchronous – wait up to 60 s for output
    }
    payload = {
        "input": {
            "image": image_data_uri,
            "prompt": user_prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", "1500")),
            "temperature": 0.2,
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    # Replicate returns output as a list of strings or a single string
    output = data.get("output", "")
    if isinstance(output, list):
        text = "".join(output)
    else:
        text = str(output)

    # Replicate does not always return token counts; approximate from char length
    in_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
    out_tokens = max(1, len(text) // 4)
    return text, in_tokens, out_tokens


def call_replicate_text(
    api_token: str,
    model: str,
    system_prompt: str,
    user_message: str,
) -> tuple[str, int, int]:
    """
    Call IBM Granite 4 (text) on Replicate using the predictions endpoint.
    Returns (response_text, input_tokens, output_tokens).
    Never logs or exposes the API token.
    """
    url = f"{REPLICATE_API_URL}/{model}/predictions"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }
    payload = {
        "input": {
            "prompt": user_message,
            "system_prompt": system_prompt,
            "max_new_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", "1500")),
            "temperature": 0.3,
            "top_p": 0.9,
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    output = data.get("output", "")
    if isinstance(output, list):
        text = "".join(output)
    else:
        text = str(output)

    in_tokens = max(1, (len(system_prompt) + len(user_message)) // 4)
    out_tokens = max(1, len(text) // 4)
    return text, in_tokens, out_tokens


# ===========================================================================
# AGENT PROMPTS
# ===========================================================================

PATHOLOGIST_SYSTEM_PROMPT = """You are an expert plant pathologist and agricultural scientist with 
decades of experience diagnosing crop diseases. Your role is to carefully examine crop and plant leaf 
images to provide accurate, evidence-based disease diagnoses.

CRITICAL GUIDELINES:
- Analyze ONLY what is visible in the image. Do NOT fabricate symptoms.
- If the image is unclear, blurry, or insufficient for reliable diagnosis, explicitly state this and 
  request a clearer image. Do NOT guess.
- Clearly label this as a preliminary AI assessment, not a confirmed laboratory diagnosis.
- Provide confidence levels where appropriate (High/Medium/Low).
- List all visible symptoms and potential alternative diagnoses.
- If multiple diseases are possible, list them in order of likelihood.
- Always recommend laboratory confirmation for definitive diagnosis.

OUTPUT FORMAT (use exactly these sections):
## Disease Assessment
**Suspected Primary Disease:** [name or "Insufficient image quality for diagnosis"]
**Confidence Level:** [High/Medium/Low/Unable to assess]

## Observed Symptoms
[Bullet list of visible symptoms]

## Possible Causes
[Bullet list of likely causes: pathogen, environmental, nutrient deficiency, etc.]

## Alternative Diagnoses
[Bullet list with brief explanations]

## Assessment Notes
[Any caveats, limitations, or requests for additional information]

## Recommendation
[Whether to proceed or request clearer image / laboratory confirmation]"""


PATHOLOGIST_USER_PROMPT_TEMPLATE = """Please analyze this crop/plant leaf image for signs of disease.

{context}

Examine the image carefully for:
1. Visible symptoms (discoloration, lesions, wilting, spots, mold, rot, etc.)
2. Disease patterns and distribution
3. Possible causal agents (fungi, bacteria, viruses, pests, nutrient issues)
4. Severity of infection

Provide your assessment following the specified format. Remember: if the image quality is 
insufficient for reliable diagnosis, say so clearly rather than guessing."""


EXTENSION_OFFICER_SYSTEM_PROMPT = """You are a friendly and experienced agricultural extension officer 
helping smallholder farmers understand crop disease diagnoses and take appropriate action.

Your job is to translate the pathologist's technical diagnosis into simple, actionable, farmer-friendly 
advice that any farmer can understand and follow safely.

CRITICAL GUIDELINES:
- Write in simple, clear language that a farmer without technical background can understand.
- Be practical and specific. Give step-by-step instructions.
- For chemical treatments: ONLY recommend when appropriate, always include:
  * General product type/active ingredient (do NOT invent specific dosages)
  * Safety precautions and PPE requirements
  * Reminder to follow product label and local regulations
- Never invent pesticide dosages or make up product names.
- Always recommend consulting a local agricultural extension officer for chemical treatment guidance.
- Include both immediate actions AND long-term prevention.
- Emphasize that this is an AI advisory and professional consultation is recommended for critical cases.

OUTPUT FORMAT (use exactly these sections):
## Farmer Advisory Report

### 🚨 Immediate Actions (Do This Now)
[Numbered list of urgent steps]

### 💊 Treatment Options
[Organic/cultural options first, then chemical if appropriate]
**Important:** Always read and follow product labels. Wear protective equipment.

### 🌱 Preventive Measures
[Long-term prevention strategies]

### 🌿 Crop-Care Recommendations
[General good practices for this crop]

### ⚠️ Safety Warnings
[Pesticide safety, PPE, storage, disposal]

### 📞 When to Seek Expert Help
[Conditions that require professional intervention]

### Summary
[2-3 sentence plain-English summary for the farmer]"""


# ===========================================================================
# MAIN DIAGNOSIS WORKFLOW
# ===========================================================================

if diagnose_btn and uploaded_file is not None:

    guardrails: GuardrailOrchestrator = st.session_state.guardrails
    session: SessionMetrics = create_session(f"session_{uuid.uuid4().hex[:8]}")
    st.session_state.session_metrics = session

    # ── Step 0: Validate API token ────────────────────────────────────────────
    key_errors: list[str] = []

    replicate_check = guardrails.tool_g.validate_api_authorization(
        replicate_api_token, "Replicate"
    )
    if not replicate_check.passed:
        key_errors.append(replicate_check.message)

    if key_errors:
        for err in key_errors:
            st.error(f"🔑 {err}")
        st.info("Please enter your Replicate API token in the sidebar and try again.")
        st.stop()

    # ── Step 1: Validate image upload ─────────────────────────────────────────
    image_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type or "image/jpeg"

    img_valid, img_results = guardrails.validate_image_upload(
        image_bytes, mime_type, uploaded_file.name
    )
    session.guardrail_results.extend([r.to_dict() for r in img_results])

    if not img_valid:
        for msg in guardrails.get_blocking_messages(img_results):
            st.error(f"🚫 Image validation failed: {msg}")
        st.stop()

    # ── Step 2: Validate additional context input ─────────────────────────────
    if additional_context.strip():
        ctx_valid, ctx_results = guardrails.validate_user_input(additional_context)
        session.guardrail_results.extend([r.to_dict() for r in ctx_results])
        if not ctx_valid:
            for msg in guardrails.get_blocking_messages(ctx_results):
                st.error(f"🚫 Input validation failed: {msg}")
            st.stop()
        for warn in guardrails.get_warning_messages(ctx_results):
            st.warning(f"⚠️ {warn}")

    # ── Step 3: Start operational monitoring ──────────────────────────────────
    guardrails.ops_g.start_session()

    # ============================================================
    # AGENT 1: PATHOLOGIST AGENT
    # ============================================================
    st.markdown("---")
    st.subheader("🔬 Agent 1: Pathologist Agent")

    path_run = start_agent_run(session, "pathologist_agent")

    with st.status("🔬 Pathologist Agent analyzing crop image...", expanded=True) as status:
        try:
            # Validate tool call
            tool_check = guardrails.tool_g.validate_tool_call(
                "replicate_vision_api", {"model": vision_model}
            )
            if not tool_check.passed:
                st.error(f"🚫 Tool access denied: {tool_check.message}")
                finish_agent_run(path_run, False, error=tool_check.message)
                st.stop()

            # Check iteration and time limits
            iter_check = guardrails.ops_g.check_iteration_limit()
            time_check = guardrails.ops_g.check_execution_time()
            if not iter_check.passed or not time_check.passed:
                st.error("🚫 Operational limit exceeded. Please try again later.")
                st.stop()

            st.write("📸 Encoding image...")
            image_b64 = encode_image_base64(image_bytes)

            context_str = (
                f"Additional context from farmer: {additional_context.strip()}"
                if additional_context.strip()
                else "No additional context provided."
            )
            user_prompt = PATHOLOGIST_USER_PROMPT_TEMPLATE.format(context=context_str)

            st.write("🔍 Sending image to IBM Granite Vision (Replicate) for analysis...")
            path_run.input_text = user_prompt

            diagnosis_text, in_tok, out_tok = call_replicate_vision(
                api_token=replicate_api_token,
                image_b64=image_b64,
                mime_type=mime_type,
                model=vision_model,
                system_prompt=PATHOLOGIST_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            path_run.input_tokens = in_tok
            path_run.output_tokens = out_tok
            path_run.output_text = diagnosis_text

            # Track cost
            cost_check = guardrails.ops_g.track_tokens_and_cost(in_tok, out_tok)
            session.guardrail_results.append(cost_check.to_dict())
            if not cost_check.passed:
                st.warning(f"⚠️ {cost_check.message}")

            st.write("✅ Vision analysis complete. Validating output...")

            # Validate output
            out_valid, out_results = guardrails.validate_agent_output(
                diagnosis_text, "pathologist_agent"
            )
            session.guardrail_results.extend([r.to_dict() for r in out_results])

            if not out_valid:
                for msg in guardrails.get_blocking_messages(out_results):
                    st.error(f"🚫 Output guardrail blocked response: {msg}")
                finish_agent_run(path_run, False, error="Output blocked by guardrail")
                st.stop()

            for warn in guardrails.get_warning_messages(out_results):
                st.warning(f"⚠️ {warn}")

            finish_agent_run(path_run, True, diagnosis_text, in_tok, out_tok)
            st.session_state.diagnosis_result = diagnosis_text
            status.update(label="✅ Pathologist Agent: Diagnosis complete!", state="complete")

        except requests.exceptions.HTTPError as e:
            error_msg = f"Replicate API error: {e.response.status_code} – {e.response.text[:200]}"
            st.error(f"🚫 {error_msg}")
            finish_agent_run(path_run, False, error=error_msg)
            st.stop()
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Replicate API. Check your internet connection."
            st.error(f"🌐 {error_msg}")
            finish_agent_run(path_run, False, error=error_msg)
            st.stop()
        except Exception as e:
            error_msg = f"Unexpected error in Pathologist Agent: {type(e).__name__}"
            st.error(f"❌ {error_msg}. Please try again.")
            finish_agent_run(path_run, False, error=str(e))
            st.stop()

    # ── Display Pathologist Output ────────────────────────────────────────────
    if st.session_state.diagnosis_result:
        with st.expander("🔬 Pathologist Agent – Full Diagnosis Report", expanded=True):
            st.markdown(st.session_state.diagnosis_result)
            st.caption(
                f"Model: `{vision_model}` | "
                f"Tokens: {path_run.input_tokens} in / {path_run.output_tokens} out | "
                f"Latency: {path_run.latency_ms:.0f} ms"
            )

    # ============================================================
    # AGENT HANDOFF: Pathologist → Extension Officer
    # ============================================================
    st.markdown("---")
    st.subheader("🔄 Agent Handoff: Pathologist → Extension Officer")

    handoff_valid, handoff_results = guardrails.validate_agent_handoff(
        "pathologist_agent",
        "extension_officer_agent",
        st.session_state.diagnosis_result or "",
    )
    session.guardrail_results.extend([r.to_dict() for r in handoff_results])

    if not handoff_valid:
        for msg in guardrails.get_blocking_messages(handoff_results):
            st.error(f"🚫 Agent handoff failed: {msg}")
        st.stop()

    st.success("✅ Diagnosis report successfully handed off to Extension Officer Agent.")

    # ============================================================
    # AGENT 2: EXTENSION OFFICER AGENT
    # ============================================================
    st.markdown("---")
    st.subheader("👨‍🌾 Agent 2: Extension Officer Agent")

    ext_run = start_agent_run(session, "extension_officer_agent")
    ext_run.input_text = st.session_state.diagnosis_result or ""

    # Build user message incorporating pathologist's diagnosis
    ext_user_message = f"""A crop has been diagnosed by the Pathologist Agent. 
Here is the full diagnostic report:

---
{st.session_state.diagnosis_result}
---

Based on this diagnosis, please generate a comprehensive, farmer-friendly advisory report 
following your defined output format. Write in simple language that any farmer can understand.
Include immediate actions, treatment options, preventive measures, and safety warnings.
If the diagnosis was uncertain or the image was unclear, recommend consulting a local 
agricultural extension officer or plant pathology laboratory."""

    with st.status("👨‍🌾 Extension Officer Agent preparing advisory report...", expanded=True) as status:
        try:
            # Validate tool call
            tool_check2 = guardrails.tool_g.validate_tool_call(
                "replicate_text_api", {"model": text_model}
            )
            if not tool_check2.passed:
                st.error(f"🚫 Tool access denied: {tool_check2.message}")
                finish_agent_run(ext_run, False, error=tool_check2.message)
                st.stop()

            iter_check2 = guardrails.ops_g.check_iteration_limit()
            time_check2 = guardrails.ops_g.check_execution_time()
            if not iter_check2.passed or not time_check2.passed:
                st.error("🚫 Operational limit exceeded.")
                st.stop()

            st.write("📝 Sending diagnosis to IBM Granite 4 (Replicate) for advisory...")

            advisory_text, ext_in_tok, ext_out_tok = call_replicate_text(
                api_token=replicate_api_token,
                model=text_model,
                system_prompt=EXTENSION_OFFICER_SYSTEM_PROMPT,
                user_message=ext_user_message,
            )

            ext_run.input_tokens = ext_in_tok
            ext_run.output_tokens = ext_out_tok
            ext_run.output_text = advisory_text

            cost_check2 = guardrails.ops_g.track_tokens_and_cost(ext_in_tok, ext_out_tok)
            session.guardrail_results.append(cost_check2.to_dict())

            st.write("✅ Advisory generated. Validating output...")

            # Validate Extension Officer output
            ext_out_valid, ext_out_results = guardrails.validate_agent_output(
                advisory_text, "extension_officer_agent"
            )
            session.guardrail_results.extend([r.to_dict() for r in ext_out_results])

            if not ext_out_valid:
                for msg in guardrails.get_blocking_messages(ext_out_results):
                    st.error(f"🚫 Output guardrail blocked advisory: {msg}")
                finish_agent_run(ext_run, False, error="Output blocked by guardrail")
                st.stop()

            for warn in guardrails.get_warning_messages(ext_out_results):
                st.warning(f"⚠️ {warn}")

            # Add mandatory disclaimer
            advisory_with_disclaimer = guardrails.hitl_g.add_disclaimer(advisory_text)

            finish_agent_run(ext_run, True, advisory_text, ext_in_tok, ext_out_tok)
            st.session_state.advisory_result = advisory_with_disclaimer
            status.update(label="✅ Extension Officer Agent: Advisory complete!", state="complete")

        except requests.exceptions.HTTPError as e:
            error_msg = f"Replicate API error: {e.response.status_code} – {e.response.text[:200]}"
            st.error(f"🚫 {error_msg}")
            finish_agent_run(ext_run, False, error=error_msg)
            st.stop()
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Replicate API. Check your internet connection."
            st.error(f"🌐 {error_msg}")
            finish_agent_run(ext_run, False, error=error_msg)
            st.stop()
        except Exception as e:
            error_msg = f"Unexpected error in Extension Officer Agent: {type(e).__name__}"
            st.error(f"❌ {error_msg}. Please try again.")
            finish_agent_run(ext_run, False, error=str(e))
            st.stop()

    # ── Display Advisory Output ───────────────────────────────────────────────
    if st.session_state.advisory_result:
        with st.expander("👨‍🌾 Extension Officer Agent – Farmer Advisory Report", expanded=True):
            st.markdown(st.session_state.advisory_result)
            st.caption(
                f"Model: `{text_model}` | "
                f"Tokens: {ext_run.input_tokens} in / {ext_run.output_tokens} out | "
                f"Latency: {ext_run.latency_ms:.0f} ms"
            )

    # ============================================================
    # COMBINED FINAL REPORT
    # ============================================================
    if st.session_state.diagnosis_result and st.session_state.advisory_result:
        st.markdown("---")
        st.subheader("📋 Combined Final Report")

        final_report = f"""# Agri Doctor – Crop Disease Management Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
Session ID: {session.session_id}

---

## PART 1: PATHOLOGIST AGENT – Disease Diagnosis
{st.session_state.diagnosis_result}

---

## PART 2: EXTENSION OFFICER AGENT – Farmer Advisory
{st.session_state.advisory_result}

---

*This report was generated by Agri Doctor using AI technology. It is a preliminary 
assessment only and is NOT a substitute for professional agricultural diagnosis. 
Always consult a certified agricultural extension officer or plant pathologist before 
taking action. Follow all local agricultural regulations and pesticide safety guidelines.*
"""
        with st.expander("📋 Full Combined Report", expanded=False):
            st.markdown(final_report)

        # Download button
        st.download_button(
            label="⬇️ Download Advisory Report (.txt)",
            data=final_report.encode("utf-8"),
            file_name=f"agri_doctor_report_{session.session_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.success(
            "✅ **Multi-Agent Workflow Complete!** Both agents have successfully "
            "collaborated to provide your crop disease advisory."
        )

    # ── Store guardrail audit in session ──────────────────────────────────────
    session.audit_log = guardrails.audit_log.copy()

    # ============================================================
    # PERFORMANCE DASHBOARD
    # ============================================================
    render_dashboard(session)


# ===========================================================================
# DISPLAY PREVIOUS RESULTS (if page reloads)
# ===========================================================================
elif (
    not diagnose_btn
    and st.session_state.diagnosis_result
    and st.session_state.advisory_result
    and st.session_state.session_metrics
):
    st.info("🔄 Showing results from previous diagnosis. Upload a new image and click Diagnose to start again.")

    with st.expander("🔬 Previous Diagnosis", expanded=False):
        st.markdown(st.session_state.diagnosis_result)

    with st.expander("👨‍🌾 Previous Advisory", expanded=False):
        st.markdown(st.session_state.advisory_result)

    # Re-render previous dashboard
    render_dashboard(st.session_state.session_metrics)

elif not diagnose_btn and uploaded_file is None:
    # Landing state
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**Step 1**\n\nEnter your Replicate API token in the sidebar.")
    with col_b:
        st.info("**Step 2**\n\nUpload a clear image of your affected crop or plant leaf.")
    with col_c:
        st.info("**Step 3**\n\nClick **Diagnose Crop** to start the two-agent AI workflow.")

    st.markdown(
        """
        ### How it works
        1. **Pathologist Agent** analyzes your crop image using **IBM Granite Vision**
           (via Replicate) to identify diseases, symptoms, and possible causes.
        2. The diagnosis is **automatically handed off** to the Extension Officer Agent.
        3. **Extension Officer Agent** uses **IBM Granite 4** (via Replicate) to generate
           a practical, farmer-friendly advisory with treatment and prevention recommendations.
        4. A full **Performance Dashboard** displays evaluation metrics for both agents.
        
        ### Safety Features
        - Comprehensive input & output guardrails
        - API token never stored or logged
        - AI disclaimer on all outputs
        - Human escalation triggers for critical cases
        - Multi-agent security controls
        """
    )
