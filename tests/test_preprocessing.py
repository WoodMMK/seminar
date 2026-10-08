"""
Unit tests for remove_lines_text_and_binarize (Dotted-Line Detection & Inpainting).
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from src.ocr_engine import remove_lines_text_and_binarize


def create_synthetic_test_image_with_dotted_line():
    """Creates a white page with simulated text and an explicit dotted line."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # 1. Add some text
    cv2.putText(img, "ลงชื่อ ......................................... ผู้เบิก", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 2. Add an explicit dotted line:
    for x in range(160, 500, 8):
        cv2.circle(img, (x, 200), 2, (0, 0, 0), -1)

    return img


def test_remove_lines_text_and_binarize_default_return():
    img = create_synthetic_test_image_with_dotted_line()
    res = remove_lines_text_and_binarize(img, thresh_bin=200, return_intermediates=False)
    assert isinstance(res, np.ndarray)
    assert len(res.shape) == 2
    assert res.shape == (600, 800)
    # Check that it's binary (only 0 and 255 values)
    unique_vals = set(np.unique(res))
    assert unique_vals.issubset({0, 255})


def test_remove_lines_text_and_binarize_intermediates():
    img = create_synthetic_test_image_with_dotted_line()
    res = remove_lines_text_and_binarize(
        img,
        thresh_bin=200,
        return_intermediates=True,
        hough_threshold=50
    )
    assert isinstance(res, dict)
    for key in [
        "final_image", "original_bgr", "gray",
        "canny_edges", "line_mask",
        "inpainted_bgr", "inpainted_gray", "final_binarized", "stats"
    ]:
        assert key in res, f"Missing key {key} in results"

    stats = res["stats"]
    assert stats["width"] == 800
    assert stats["height"] == 600
    assert stats["thresh_bin"] == 200
    assert "line_mask_pixels" in stats
    assert "inpainted_pixels" in stats


def test_pil_and_path_inputs(tmp_path):
    img_bgr = create_synthetic_test_image_with_dotted_line()
    img_path = tmp_path / "test_doc.png"
    cv2.imwrite(str(img_path), img_bgr)

    # 1. Test from Path
    res_path = remove_lines_text_and_binarize(img_path, thresh_bin=200)
    assert isinstance(res_path, np.ndarray)

    # 2. Test from PIL Image
    pil_img = Image.open(img_path)
    res_pil = remove_lines_text_and_binarize(pil_img, thresh_bin=200)
    assert isinstance(res_pil, np.ndarray)


if __name__ == "__main__":
    import tempfile
    print("Running preprocessing tests...")
    test_remove_lines_text_and_binarize_default_return()
    print("  [PASS] test_remove_lines_text_and_binarize_default_return")
    test_remove_lines_text_and_binarize_intermediates()
    print("  [PASS] test_remove_lines_text_and_binarize_intermediates")
    tmp = Path(tempfile.mkdtemp())
    test_pil_and_path_inputs(tmp)
    print("  [PASS] test_pil_and_path_inputs")
    print("All tests passed successfully!")
