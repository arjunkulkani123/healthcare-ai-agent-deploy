"""
Document/Image Understanding -- Orchestrator
=================================================

Ties the whole pipeline together:

    Image -> Preprocessing -> OCR -> Field Extraction -> Validation
          -> Agent-style response

This satisfies the course's image/video-processing mini-project
requirement (Section 14 of the roadmap), while staying within scope:
it extracts and organizes ADMINISTRATIVE information from a document
(appointment slips, registration forms) -- it does NOT read medical
content or make any clinical judgment.
"""

from preprocessing import load_and_preprocess
from ocr import extract_text
from field_extraction import parse_fields
from validation import validate_fields


def process_document(image_path: str) -> dict:
    """
    Args:
        image_path: path to an uploaded document image
    Returns:
        {
          "raw_text": str,
          "fields": dict,
          "validation": dict (see validation.py),
          "response": str -- agent-style natural-language summary
        }
    """
    processed_image = load_and_preprocess(image_path)
    raw_text = extract_text(processed_image)
    fields = parse_fields(raw_text)
    validation = validate_fields(fields)

    response = _build_response(fields, validation)

    return {
        "raw_text": raw_text,
        "fields": fields,
        "validation": validation,
        "response": response,
    }


def _build_response(fields: dict, validation: dict) -> str:
    if not fields:
        return (
            "I couldn't clearly read any structured information from this "
            "document. Could you try uploading a clearer photo, or type "
            "the appointment details directly?"
        )

    found_parts = []
    if "date" in fields:
        found_parts.append(f"an appointment date of {fields['date']}")
    if "facility" in fields:
        found_parts.append(f"the facility '{fields['facility']}'")
    if "registration_no" in fields:
        found_parts.append(f"a registration number ({fields['registration_no']})")
    if "service" in fields:
        found_parts.append(f"the service '{fields['service']}'")

    summary = (
        "Your uploaded document appears to contain " + ", ".join(found_parts) + ". "
        "I can help organize this information, but I can't verify medical "
        "conclusions from the document."
    )

    warnings = [c["message"] for c in validation["checks"] if c["status"] == "warning"]
    if warnings:
        summary += "\n\nA few things worth double-checking:\n" + "\n".join(f"  \u26a0 {w}" for w in warnings)

    return summary


if __name__ == "__main__":
    import tempfile
    import os
    from sample_generator import generate_appointment_slip

    tmp_dir = tempfile.gettempdir()

    # Sample 1: everything valid
    path1 = generate_appointment_slip(
        os.path.join(tmp_dir, "slip_valid.png"),
        hospital="Govt_Hospital_A", date="31/08/2026",  # a Monday
    )
    result1 = process_document(path1)
    print("=" * 70)
    print("SAMPLE 1: valid, consistent slip")
    print("=" * 70)
    print(result1["response"])

    # Sample 2: date that doesn't match the facility's open days
    path2 = generate_appointment_slip(
        os.path.join(tmp_dir, "slip_bad_date.png"),
        hospital="Govt_Hospital_A", date="30/08/2026",  # a Sunday -- hospital closed
    )
    result2 = process_document(path2)
    print("\n" + "=" * 70)
    print("SAMPLE 2: date falls on a day the facility is closed")
    print("=" * 70)
    print(result2["response"])
