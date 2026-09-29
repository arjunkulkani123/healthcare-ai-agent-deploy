"""
Image Preprocessing (for OCR)
=================================

Real phone-camera photos of documents are messy: uneven lighting, low
contrast, sometimes slightly rotated. Preprocessing before OCR
significantly improves text-extraction accuracy. Uses only Pillow (no
OpenCV dependency, to keep installation simple on Windows).

Pipeline: grayscale -> contrast boost -> upscale (if small) -> binarize
(threshold to pure black/white).
"""

from PIL import Image, ImageOps, ImageEnhance


def preprocess(image: Image.Image, target_width: int = 1200, threshold: int = 150) -> Image.Image:
    """
    Args:
        image: a PIL Image (any mode)
        target_width: upscale small images to at least this width, since
                       OCR performs poorly on low-resolution text
        threshold: grayscale cutoff (0-255) for binarization; pixels
                   darker than this become black, lighter become white
    Returns:
        A preprocessed, OCR-ready PIL Image (mode "L", black-on-white text)
    """
    # 1. Grayscale
    gray = ImageOps.grayscale(image)

    # 2. Upscale if the image is small (common with cropped phone photos)
    if gray.width < target_width:
        scale = target_width / gray.width
        new_size = (target_width, int(gray.height * scale))
        gray = gray.resize(new_size, Image.LANCZOS)

    # 3. Boost contrast so faint text becomes clearer before thresholding
    gray = ImageEnhance.Contrast(gray).enhance(1.8)

    # 4. Binarize (simple global threshold)
    binarized = gray.point(lambda p: 255 if p > threshold else 0)

    return binarized


def load_and_preprocess(path: str) -> Image.Image:
    image = Image.open(path)
    return preprocess(image)


if __name__ == "__main__":
    import tempfile
    import os
    from sample_generator import generate_appointment_slip

    tmp_dir = tempfile.gettempdir()
    sample_path = generate_appointment_slip(os.path.join(tmp_dir, "sample_slip.png"), add_noise=True)
    processed = load_and_preprocess(sample_path)
    out_path = os.path.join(tmp_dir, "sample_slip_preprocessed.png")
    processed.save(out_path)
    print(f"Preprocessed image saved to: {out_path}")
