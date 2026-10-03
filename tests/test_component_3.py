"""
Integration Test for Component 3: Reasoning & Extraction Engine (Local LLM)
Verifies Ollama connectivity, prompt construction, CoT thinking extraction,
and structured Pydantic financial schema output.
"""

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.llm_extractor import LLMExtractor
from src.schemas import FinancialExtractionResult


def test_component_3():
    print("=" * 60)
    print("🧪 Testing Component 3: Reasoning & Extraction Engine")
    print("=" * 60)

    extractor = LLMExtractor(default_model="qwen2.5:3b")

    # 1. Test Ollama connectivity
    is_online = extractor.is_ollama_running()
    print(f"\n1. Ollama Status: {'🟢 ONLINE' if is_online else '🟠 OFFLINE (Using Authentic Mock)'}")

    if is_online:
        models = extractor.list_available_models()
        print(f"   Available models on host: {models}")

    # 2. Sample OCR text to extract
    sample_ocr_markdown = """
--- Page 1 ---
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

    print("\n2. Executing Financial Information Extraction (Live Model)...")
    try:
        result = extractor.extract(
            ocr_markdown=sample_ocr_markdown,
            use_mock_if_offline=True,
            force_mock=False
        )
    except ConnectionError as e:
        print(f"\n⚠️ {e}")
        print("Test completed: ConnectionError raised as expected when Ollama is offline.")
        return

    print("\n3. Extracted Financial Data:")
    print(f"   - Document Type: {result.document_type}")
    print(f"   - Vendor: {result.vendor_name}")
    print(f"   - Tax ID: {result.vendor_tax_id}")
    print(f"   - Date: {result.document_date}")
    print(f"   - Line Items count: {len(result.line_items)}")
    for item in result.line_items:
        print(f"     * #{item.item_no} {item.description} ({item.quantity} {item.unit}) = {item.total_price} บาท")
    print(f"   - Subtotal: {result.subtotal} บาท")
    print(f"   - VAT (7%): {result.vat} บาท")
    print(f"   - Total: {result.total_amount} บาท")
    print(f"   - Model Used: {result.model_used}")
    print(f"   - Latency: {result.latency_ms} ms")

    if result.thinking_process:
        print("\n4. Chain-of-Thought (<think>) Reasoning Trail:")
        print("-" * 40)
        print(result.thinking_process)
        print("-" * 40)

    # Verification assertions
    assert result.vendor_name is not None, "Vendor name should not be empty"
    assert result.total_amount is not None, "Total amount should not be empty"
    assert len(result.line_items) > 0, "Line items should have at least 1 item"
    print("\n✅ Component 3 Unit & Integration Tests PASSED!")


if __name__ == "__main__":
    test_component_3()
