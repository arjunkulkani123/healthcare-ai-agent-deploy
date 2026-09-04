"""
LLM-based NLU (real upgrade over nlu.py's regex approach)
==============================================================

This replaces pattern-matching with an actual call to Claude, so the
agent can understand genuinely varied phrasing instead of only the
specific keywords nlu.py looks for.

Requires an Anthropic API key. Set it as:
    - Local development: environment variable ANTHROPIC_API_KEY
    - Streamlit Cloud: add ANTHROPIC_API_KEY in the app's "Secrets" settings
      (Settings -> Secrets, NOT committed to GitHub)

If no key is available, or the API call fails for any reason, this
module returns None and the caller (supervisor_agent.py) falls back to
the regex-based nlu.py -- so the app never breaks, it just gets less
sophisticated at understanding phrasing.
"""

import os
import json

SYSTEM_PROMPT = """You are the natural-language-understanding layer for a healthcare service-navigation assistant. Extract structured facts from the user's request. Output ONLY valid JSON, nothing else, matching this exact schema:

{
  "expert_facts": {
    "age": <int or omit>,
    "has_fever": <bool>,
    "fever_severity": "<none|mild|high>",
    "fever_duration_days": <int>,
    "government_preference": <bool, omit if not mentioned>,
    "budget_constrained": <bool, omit if not mentioned>,
    "needs_vaccination": <bool, omit if not mentioned>,
    "chronic_condition": <bool, omit if not mentioned>,
    "mobility_impaired": <bool, omit if not mentioned>
  },
  "scheduling_overrides": {
    "preferred_time_of_day": "<morning|afternoon|evening, omit if not mentioned>",
    "max_distance_km": <number, omit if not mentioned>,
    "facility_preference": "<public|private, omit if not mentioned>"
  },
  "assumptions": ["<plain-English note about anything you inferred rather than were told directly>"]
}

Rules:
- Only include a field if it is stated or reasonably implied; omit fields you have no basis for.
- Never invent a diagnosis or medical conclusion -- you are extracting SERVICE-NAVIGATION facts only (age, symptoms mentioned, preferences, constraints), not making medical judgments.
- If the user says something vague like "nearby", interpret it as roughly 10 km and note that assumption in "assumptions".
- Output ONLY the JSON object, no markdown fences, no commentary."""


def extract_facts_with_llm(text: str) -> tuple:
    """
    Returns (result_dict_or_None, error_reason_or_None).

    If successful: (dict, None)
    If it fails for any reason: (None, "a short explanation of why")
    """
    api_key = _get_api_key()
    if not api_key:
        return None, "No ANTHROPIC_API_KEY found in environment variables or st.secrets."

    try:
        import anthropic
    except ImportError:
        return None, "The 'anthropic' Python package is not installed in this environment."

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)

        return {
            "expert_facts": parsed.get("expert_facts", {}),
            "scheduling_overrides": parsed.get("scheduling_overrides", {}),
            "assumptions": parsed.get("assumptions", []),
        }, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


if __name__ == "__main__":
    example = (
        "My mom's been burning up since last night, she's a senior "
        "citizen, and we're pretty broke right now -- need somewhere "
        "close by."
    )
    result, error = extract_facts_with_llm(example)
    if result is None:
        print(f"LLM unavailable: {error} -- would fall back to regex NLU.")
    else:
        print(json.dumps(result, indent=2))
