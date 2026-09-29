"""
Synthetic Document Generator (for testing/demo only)
========================================================

Since we don't have real patient documents to test with (and shouldn't
use real ones anyway), this generates a plain, printed-looking
"appointment slip" image with known ground-truth text. This lets the
OCR pipeline be tested end-to-end and its accuracy verified, before a
real user ever uploads a real photo.

In the live system, this module is NOT used -- real uploaded images go
straight into preprocessing.py / ocr.py.
"""

from PIL import Image, ImageDraw, ImageFont


def generate_appointment_slip(
    path: str,
    hospital: str = "Govt_Hospital_A",
    registration_no: str = "REG-58421",
    patient_name: str = "Ramesh Kumar",
    service: str = "OPD",
    date: str = "02/09/2026",
    time: str = "10:30",
    add_noise: bool = False,
) -> str:
    """Creates a simple white-background printed slip and saves it to `path`."""
    width, height = 640, 420
    img = Image.new("L", (width, height), color=255)  # grayscale, white background
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except Exception:
        try:
            # Windows ships Arial by default; Pillow searches the OS
            # font directory automatically for bare font names.
            font = ImageFont.truetype("arial.ttf", 20)
            font_bold = ImageFont.truetype("arialbd.ttf", 24)
        except Exception:
            # Last-resort fallback: Pillow's built-in font, but at a
            # readable size (requires Pillow >= 10.1). Without this,
            # older code fell back to a tiny bitmap font that OCR
            # struggled to read accurately -- this was the actual
            # root cause of missed fields on some machines.
            font = ImageFont.load_default(size=20)
            font_bold = ImageFont.load_default(size=24)

    lines = [
        (f"{hospital.replace('_', ' ').upper()} - APPOINTMENT SLIP", font_bold),
        ("", font),
        (f"Registration No: {registration_no}", font),
        (f"Patient Name: {patient_name}", font),
        (f"Facility: {hospital}", font),
        (f"Service: {service}", font),
        (f"Date: {date}", font),
        (f"Time: {time}", font),
    ]

    y = 30
    for text, f in lines:
        draw.text((30, y), text, fill=0, font=f)
        y += 40

    if add_noise:
        import random
        pixels = img.load()
        for _ in range(int(width * height * 0.01)):
            x = random.randint(0, width - 1)
            yy = random.randint(0, height - 1)
            pixels[x, yy] = random.choice([0, 255])

    img.save(path)
    return path


if __name__ == "__main__":
    import tempfile
    import os
    out_path = os.path.join(tempfile.gettempdir(), "sample_slip.png")
    generate_appointment_slip(out_path)
    print(f"Sample slip generated at: {out_path}")
