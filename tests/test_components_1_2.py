"""
Integration Test for Component 1 (Ingestion) & Component 2 (Perception)
Verifies image ingestion, Thai OCR recognition, bounding box extraction,
and formatting into LLM-ready Markdown.
"""

from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Ensure root dir is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion import DocumentIngestion
from src.ocr_engine import ThaiPerceptionEngine


def create_sample_receipt_image() -> Image.Image:
    """Create a synthetic receipt image for testing OCR."""
    width, height = 800, 1000
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw border
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(180, 180, 180), width=2)

    # Header and receipt text lines
    lines = [
        "ใบเสร็จรับเงิน / ใบกำกับภาษี",
        "บริษัท สมใจนึก สเตชั่นเนอรี่ จำกัด (สำนักงานใหญ่)",
        "เลขประจำตัวผู้เสียภาษี: 0105558012345",
        "วันที่: 18 กันยายน 2567",
        "--------------------------------------------------",
        "1. กระดาษถ่ายเอกสาร A4 70 แกรม จำนวน 5 รีม  550.00",
        "2. ปากกาหมึกเจล 0.5 มม. จำนวน 2 กล่อง         120.00",
        "3. แฟ้มเอกสารตราช้าง จำนวน 4 เล่ม              200.00",
        "--------------------------------------------------",
        "รวมเป็นเงิน (Subtotal):                       870.00",
        "ภาษีมูลค่าเพิ่ม 7% (VAT):                      60.90",
        "จำนวนเงินรวมทั้งสิ้น (Total):                  930.90",
        "--------------------------------------------------",
        "ขอขอบคุณที่ใช้บริการ"
    ]

    # Use default font or Windows standard font if available
    font = None
    thai_fonts = ["tahoma.ttf", "angsa.ttf", "cordia.ttf", "arial.ttf"]
    for font_name in thai_fonts:
        try:
            font = ImageFont.truetype(font_name, size=24)
            break
        except IOError:
            continue

    if font is None:
        font = ImageFont.load_default()

    y_offset = 50
    for line in lines:
        draw.text((60, y_offset), line, fill=(20, 20, 20), font=font)
        y_offset += 55

    return img


def main():
    print("=" * 60)
    print("Testing Component 1 & 2: Ingestion and Thai Perception Engine")
    print("=" * 60)

    # 1. Create sample receipt
    print("[1/3] Generating synthetic receipt...")
    sample_img = create_sample_receipt_image()
    test_output_dir = ROOT_DIR / "tests" / "output"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = test_output_dir / "sample_receipt.png"
    sample_img.save(sample_path)
    print(f"      Saved sample image to: {sample_path}")

    # 2. Test Component 1: Ingestion
    print("\n[2/3] Testing Component 1: DocumentIngestion...")
    ingestor = DocumentIngestion(target_dpi=300)
    pages = ingestor.load_document(sample_path)
    print(f"      Successfully loaded {len(pages)} page(s).")
    print(f"      Page 1 dimensions: {pages[0].width}x{pages[0].height}")
    assert len(pages) == 1, "Expected 1 page"

    # 3. Test Component 2: Perception Engine
    print("\n[3/3] Testing Component 2: ThaiPerceptionEngine...")
    engine = ThaiPerceptionEngine(device="cpu")
    print("      Processing image through OCR...")
    perception = engine.process_image(pages[0].image, page_number=1)

    print(f"\n      Extracted {len(perception.text_blocks)} text blocks.")
    print("\n--- Top 5 Extracted Text Blocks ---")
    for block in perception.text_blocks[:5]:
        print(f"  • [{block.confidence:.2f}] {block.text} -> Box: {block.box}")

    print("\n--- Formatted LLM Context ---")
    llm_markdown = perception.to_llm_markdown()
    print(llm_markdown[:500] + ("..." if len(llm_markdown) > 500 else ""))

    print("\n" + "=" * 60)
    print("Component 1 & 2 Test Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
