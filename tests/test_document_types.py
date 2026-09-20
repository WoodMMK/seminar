"""
Test suite verifying Type-Directed Extraction for all 9 university document types.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.llm_extractor import LLMExtractor
from src.schemas import DocumentType, DOCUMENT_TYPE_TITLES

SAMPLE_MEMO_OCR = """--- Page 1 ---
บันทึกข้อความ
ส่วนงาน: ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยมหิดล
ที่: อว 78.03/ว.0425  วันที่: 15 สิงหาคม 2567
เรื่อง: ขออนุมัติหลักการจัดโครงการสัมมนาเชิงปฏิบัติการวิศวกรรมปัญญาประดิษฐ์
เรียน: คณบดีคณะวิศวกรรมศาสตร์

ด้วยภาควิชาวิศวกรรมคอมพิวเตอร์ มีความประสงค์จะจัดโครงการสัมมนาเชิงปฏิบัติการ AI ทางวิศวกรรม
เพื่อเสริมสร้างทักษะนักศึกษา โดยมีประมาณการค่าใช้จ่ายดังนี้:
1. ค่าตอบแทนวิทยากรบรรยาย 6 ชั่วโมง เป็นเงิน 6,000 บาท
2. ค่าอาหารว่างและเครื่องดื่ม 50 ชุด เป็นเงิน 2,500 บาท
3. ค่าวัสดุและเอกสารประกอบ 50 ชุด เป็นเงิน 3,500 บาท
รวมเป็นเงินงบประมาณทั้งสิ้น 12,000.00 บาท

จึงเรียนมาเพื่อโปรดพิจารณาอนุมัติหลักการ
(ผู้ช่วยศาสตราจารย์ ดร.สมชาย ใจดี)
หัวหน้าภาควิชาวิศวกรรมคอมพิวเตอร์
"""


def test_all_document_types():
    print("=" * 70)
    print("🧪 Testing Type-Directed Extraction across all Document Types")
    print("=" * 70)

    extractor = LLMExtractor()

    for doc_type in DocumentType:
        type_code = doc_type.value
        type_title = DOCUMENT_TYPE_TITLES[type_code]
        print(f"\nTesting: [{type_code}] -> {type_title}")

        # Run extraction in mock mode to verify schema & dictionary mapping
        res = extractor.extract(
            ocr_markdown=SAMPLE_MEMO_OCR,
            document_type=type_code,
            force_mock=True
        )

        print(f"  ✓ Document Type: {res.document_type} ({res.document_type_name_th})")
        print(f"  ✓ Extracted Fields: {list(res.fields.keys())}")
        print(f"  ✓ Total Amount: {res.total_amount}")
        print(f"  ✓ Items count: {len(res.expense_items)}")

        assert res.document_type == type_code
        assert len(res.fields) > 0
        if type_code != DocumentType.PARCEL_INSPECTION.value:
            assert res.total_amount is not None, f"Total amount should not be None for {type_code}"

    print("\n" + "=" * 70)
    print("✅ All 10 Document Types passed schema and extraction verification!")
    print("=" * 70)


if __name__ == "__main__":
    test_all_document_types()
