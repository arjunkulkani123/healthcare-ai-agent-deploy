"""
Field Extraction
====================

Parses raw OCR text into structured fields. Uses simple, robust regex
patterns matched against "Label: value" style lines, which is how most
printed appointment slips / registration forms are formatted.

Scope reminder: this extracts ADMINISTRATIVE information only
(facility, date, time, registration number, service name) -- never
attempts to read or interpret medical/clinical content from a document.
"""

import re

FIELD_PATTERNS = {
    "registration_no": r"Registration\s*No\.?[:\-]?\s*([A-Za-z0-9\-]+)",
    "facility": r"Facility[:\-]?\s*([A-Za-z0-9_ ]+)",
    "service": r"Service[:\-]?\s*([A-Za-z0-9_ ]+)",
    "date": r"Date[:\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
    "time": r"Time[:\-]?\s*(\d{1,2}:\d{2})",
}


def parse_fields(raw_text: str) -> dict:
    """
    Args:
        raw_text: text extracted by ocr.py
    Returns:
        dict with whatever fields were found (missing fields are simply
        absent from the dict, not filled with placeholder values -- the
        caller should treat absence as "this document didn't state it"
        rather than guessing).
    """
    fields = {}
    for field_name, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            fields[field_name] = match.group(1).strip()
    return fields


if __name__ == "__main__":
    import tempfile
    import os
    from ocr import extract_text
    from preprocessing import load_and_preprocess
    from sample_generator import generate_appointment_slip

    sample_path = generate_appointment_slip(os.path.join(tempfile.gettempdir(), "sample_slip.png"))
    processed = load_and_preprocess(sample_path)
    raw_text = extract_text(processed)

    fields = parse_fields(raw_text)
    print("Parsed fields:")
    for k, v in fields.items():
        print(f"  {k}: {v}")
