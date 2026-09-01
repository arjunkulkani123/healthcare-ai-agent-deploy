"""
Healthcare Experience Agent -- Streamlit Dashboard
=======================================================

The UI described in the roadmap's Section 18: a request box, a live
"agent trace" panel showing the reasoning steps, a recommendation with
explanation, and a document-upload flow for the OCR pipeline.

Run with:
    streamlit run app/frontend/app.py
"""

import sys
import os

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "agents"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "ai", "vision"))

import streamlit as st
from supervisor_agent import handle_request

st.set_page_config(page_title="Healthcare Experience Agent", page_icon="\u2695", layout="wide")

# ---------------------------------------------------------------------
# Styling -- calm clinical palette (deep teal + warm off-white), not the
# default Streamlit look. Kept restrained: one accent color, generous
# spacing, no decoration that doesn't serve the content.
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --teal-900: #0b3d3a;
            --teal-700: #0f6b63;
            --teal-500: #14919b;
            --cream: #f7f5f0;
            --ink: #1c2521;
        }
        .stApp { background-color: var(--cream); }
        h1, h2, h3 { color: var(--teal-900) !important; font-family: 'Georgia', serif; }
        .subtitle { color: #5a6b66; font-size: 1.05rem; margin-top: -0.6rem; margin-bottom: 1.5rem; }
        .trace-step {
            padding: 0.5rem 0.8rem;
            border-left: 3px solid var(--teal-500);
            background: #ffffff;
            margin-bottom: 0.4rem;
            border-radius: 0 6px 6px 0;
            font-size: 0.92rem;
            color: var(--ink);
        }
        .result-card {
            background: #ffffff;
            border: 1px solid #e2e2dc;
            border-radius: 10px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
        }
        .badge {
            display: inline-block;
            padding: 0.15rem 0.7rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .badge-high { background: #fde2e1; color: #9c2b26; }
        .badge-medium { background: #fdf0d0; color: #8a5e10; }
        .badge-routine { background: #dcf0e6; color: #1f6b46; }
        .scope-note {
            font-size: 0.85rem;
            color: #78877f;
            border-top: 1px solid #e2e2dc;
            padding-top: 0.7rem;
            margin-top: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# Healthcare Experience Agent")
st.markdown(
    '<p class="subtitle">Describe what you need in plain language -- '
    "the agent classifies urgency, schedules a real appointment slot, "
    "and plans a route, explaining every step.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------
EXAMPLES = {
    "Urgent senior, budget-conscious": (
        "My mother has had a high fever since yesterday and I need to "
        "find a government hospital nearby. She is 62 and we don't have "
        "much money."
    ),
    "Routine vaccination": (
        "I'm 28 years old and need a vaccination appointment tomorrow "
        "morning, preferably at a government facility nearby."
    ),
    "Chronic condition, mobility-impaired": (
        "My father is 70 and has a chronic condition, he needs a "
        "specialist consultation. He has trouble walking so we need "
        "somewhere within 8 km."
    ),
}

def _use_example(text: str):
    st.session_state["user_text"] = text


col_input, col_examples = st.columns([3, 1])
with col_input:
    user_text = st.text_area(
        "How can I help you?",
        height=100,
        placeholder="e.g. My mother has had a high fever since yesterday and I need to find a government hospital nearby...",
        key="user_text",
    )
with col_examples:
    st.markdown("**Try an example**")
    for label, text in EXAMPLES.items():
        st.button(label, use_container_width=True, on_click=_use_example, args=(text,))

run_clicked = st.button("Ask Agent", type="primary")

# ---------------------------------------------------------------------
# Run the agent and display results
# ---------------------------------------------------------------------
if run_clicked and st.session_state.get("user_text", "").strip():
    with st.spinner("Reasoning through the request..."):
        result = handle_request(st.session_state["user_text"])

    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("### Agent Trace")
        for step in result["trace"]:
            st.markdown(f'<div class="trace-step">{step}</div>', unsafe_allow_html=True)

    with right:
        st.markdown("### Recommendation")

        urgency = result["expert_result"]["urgency_level"] or "routine"
        badge_class = {"high": "badge-high", "medium": "badge-medium"}.get(urgency, "badge-routine")

        st.markdown(
            f"""
            <div class="result-card">
                <span class="badge {badge_class}">{urgency.upper()} URGENCY</span>
                <span class="badge badge-routine">{result['expert_result']['recommended_service'].replace('_', ' ')}</span>
                <p style="margin-top:0.8rem; white-space: pre-line;">{result['expert_result']['explanation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["appointment"]:
            appt = result["appointment"]
            st.markdown(
                f"""
                <div class="result-card">
                    <h4 style="margin-top:0;">Appointment</h4>
                    <p><b>{appt['doctor']}</b> at <b>{appt['hospital']}</b><br>
                    {appt['date']} at {appt['time']}</p>
                    <p style="white-space: pre-line; font-size: 0.9rem;">{result['appointment_explanation']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result["route"] and result["route"]["path"]:
                route = result["route"]
                path_str = " &rarr; ".join(route["path"])
                st.markdown(
                    f"""
                    <div class="result-card">
                        <h4 style="margin-top:0;">Suggested Route</h4>
                        <p>{path_str}</p>
                        <p style="color:#5a6b66; font-size:0.85rem;">
                            Approx. {route['cost']} km &middot; A* expanded {route['nodes_expanded']} nodes
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                "No appointment slot satisfies all constraints. Try relaxing "
                "the time window or distance limit."
            )

        st.markdown(
            '<p class="scope-note">This is a service-navigation recommendation, '
            "not a medical diagnosis. Please consult a qualified clinician for "
            "any medical concerns.</p>",
            unsafe_allow_html=True,
        )

elif run_clicked:
    st.warning("Please describe what you need first.")

# ---------------------------------------------------------------------
# Document upload (OCR pipeline)
# ---------------------------------------------------------------------
st.markdown("---")
with st.expander("\U0001F4CE Upload a document (appointment slip, registration form)"):
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            from document_processor import process_document

            with st.spinner("Reading document..."):
                doc_result = process_document(tmp_path)

            st.image(uploaded_file, width=300)
            st.markdown("**Agent response:**")
            st.write(doc_result["response"])

            with st.expander("Raw extracted fields"):
                st.json(doc_result["fields"])
        except Exception as e:
            st.error(
                f"Couldn't process this document: {e}\n\n"
                "Make sure Tesseract OCR is installed and configured in "
                "ai/vision/ocr.py (see the comment at the top of that file)."
            )
