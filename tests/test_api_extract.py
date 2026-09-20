import json
import urllib.request

def test_api():
    print("Testing GET /api/llm/status...")
    with urllib.request.urlopen("http://localhost:8000/api/llm/status", timeout=5) as resp:
        status_data = json.loads(resp.read().decode())
    print("Status:", json.dumps(status_data, ensure_ascii=False, indent=2))

    sample_ocr = """--- Page 1 ---
### Extracted Text Content:
ใบเสร็จรับเงิน / ใบกำกับภาษี
บริษัท สมใจนึก สเตชั่นเนอรี่ จำกัด (สำนักงานใหญ่)
เลขประจำตัวผู้เสียภาษี: 0105558012345
วันที่: 18 กันยายน 2567
1. กระดาษถ่ายเอกสาร A4 70 แกรม จำนวน 5 รีม  550.00
2. ปากกาหมึกเจล 0.5 มม. จำนวน 2 กล่อง         120.00
3. แฟ้มเอกสารตราช้าง จำนวน 4 เล่ม              200.00
รวมเป็นเงิน (Subtotal):                       870.00
ภาษีมูลค่าเพิ่ม 7% (VAT):                      60.90
จำนวนเงินรวมทั้งสิ้น (Total):                  930.90
ขอขอบคุณที่ใช้บริการ
"""

    print("\nTesting POST /api/llm/extract with force_mock=True (Using Real Ollama Snapshot)...")
    payload = json.dumps({"ocr_markdown": sample_ocr, "force_mock": True}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/api/llm/extract",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        extract_mock = json.loads(resp.read().decode())
    
    print(f"Mock Success!")
    print(f"  - Vendor: {extract_mock['vendor_name']}")
    print(f"  - Subtotal: {extract_mock['subtotal']}")
    print(f"  - Total: {extract_mock['total_amount']}")
    print(f"  - Items count: {len(extract_mock['line_items'])}")
    print(f"  - Is Mock: {extract_mock['is_mock']}")
    print(f"  - Model: {extract_mock['model_used']}")
    print(f"  - Validation Status: {extract_mock.get('validation', {}).get('status')}")

    assert extract_mock["vendor_name"] == "บริษัท สมใจนึก สเตชั่นเนอรี่ จำกัด (สำนักงานใหญ่)"
    assert extract_mock["total_amount"] == 930.9
    assert extract_mock["is_mock"] is True
    assert "validation" in extract_mock, "Validation report should be included in extraction output"
    assert extract_mock["validation"]["status"] in ["PASSED", "WARNING", "ERROR"]

    print("\nTesting POST /api/validate (Standalone Validation Endpoint)...")
    val_payload = json.dumps(extract_mock).encode("utf-8")
    req_val = urllib.request.Request(
        "http://localhost:8000/api/validate",
        data=val_payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req_val, timeout=5) as resp:
        standalone_val = json.loads(resp.read().decode())
    print("  - Standalone validation result:", standalone_val["status"])
    assert "is_valid" in standalone_val
    assert "math_report" in standalone_val

    print("\nTesting POST /api/llm/extract with force_mock=False (Live Ollama Inference)...")
    payload_real = json.dumps({"ocr_markdown": sample_ocr, "model_name": "qwen2.5:3b", "force_mock": False}).encode("utf-8")
    req_real = urllib.request.Request(
        "http://localhost:8000/api/llm/extract",
        data=payload_real,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req_real, timeout=60) as resp:
        extract_real = json.loads(resp.read().decode())

    print(f"Live Ollama Inference Success!")
    print(f"  - Vendor: {extract_real['vendor_name']}")
    print(f"  - Subtotal: {extract_real['subtotal']}")
    print(f"  - Total: {extract_real['total_amount']}")
    print(f"  - Items count: {len(extract_real['line_items'])}")
    print(f"  - Is Mock: {extract_real['is_mock']}")
    print(f"  - Latency: {extract_real['latency_ms']} ms")
    print(f"  - Model: {extract_real['model_used']}")
    print(f"  - Validation Status: {extract_real.get('validation', {}).get('status')}")

    assert extract_real["is_mock"] is False
    assert extract_real["total_amount"] == 930.9
    assert "validation" in extract_real
    assert extract_real["validation"]["is_valid"] is True

    print("\n✅ API verification test passed!")

if __name__ == "__main__":
    test_api()
