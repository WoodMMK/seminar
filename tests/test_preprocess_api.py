"""
Tests for Preprocessing UI and API endpoints (/api/preprocess/*).
"""

from fastapi.testclient import TestClient
from pathlib import Path
from src.app import app

client = TestClient(app)


def test_ui_routes():
    """Verify that /ui, /preprocess and / return HTML."""
    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "text/html" in res_ui.headers.get("content-type", "")
    assert "Thai OCR Dotted Line Inpainting Studio" in res_ui.text

    res_pre = client.get("/preprocess")
    assert res_pre.status_code == 200

    # Browser request to /
    res_root = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert res_root.status_code == 200
    assert "Thai OCR Dotted Line Inpainting Studio" in res_root.text


def test_samples_endpoint():
    """Verify that /api/preprocess/samples returns a non-empty list of samples."""
    res = client.get("/api/preprocess/samples")
    assert res.status_code == 200
    samples = res.json()
    assert isinstance(samples, list)
    assert len(samples) > 0
    first = samples[0]
    assert "folder" in first
    assert "name" in first
    assert "path" in first


def test_process_endpoint_with_sample():
    """Verify that /api/preprocess/process returns all 6 image dataurls and stats."""
    res_samples = client.get("/api/preprocess/samples")
    samples = res_samples.json()
    sample_path = samples[0]["path"]

    res = client.post(
        "/api/preprocess/process",
        data={
            "sample_path": sample_path,
            "thresh_bin": 200,
            "hough_threshold": 150,
            "line_thickness": 7,
            "horizontal_only": "true"
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data.get("success") is True
    images = data.get("images", {})
    for expected_key in [
        "original", "gray", "canny_edges", "line_mask", "inpainted", "final_binarized"
    ]:
        assert expected_key in images
        assert images[expected_key].startswith("data:image/png;base64,")

    stats = data.get("stats", {})
    assert stats.get("thresh_bin") == 200
    assert stats.get("width") > 0
    assert stats.get("height") > 0


if __name__ == "__main__":
    print("Testing Preprocess API...")
    test_ui_routes()
    print("  [PASS] test_ui_routes")
    test_samples_endpoint()
    print("  [PASS] test_samples_endpoint")
    test_process_endpoint_with_sample()
    print("  [PASS] test_process_endpoint_with_sample")
    print("All Preprocess API tests passed!")
