"""
Component 3: Reasoning & Extraction Engine (Local LLM via Ollama)
Extracts structured financial data for 9 specific university document types
as well as general commercial receipts.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.schemas import (
    DOCUMENT_TYPE_TITLES,
    DocumentType,
    ExpenseItem,
    FinancialExtractionResult,
    LineItem,
    TypeDirectedExtractionResult,
)
from src.validator import FinancialDocumentValidator

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
REAL_RESPONSE_FIXTURE = FIXTURES_DIR / "real_ollama_response.json"


PROMPT_CONFIGS: Dict[str, Dict[str, Any]] = {
    DocumentType.PRINCIPLE_APPROVAL_REQUEST.value: {
        "title": "1. เอกสารขออนุมัติหลักการ",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'เอกสารขออนุมัติหลักการ' สำหรับการใช้จ่ายงบประมาณ\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร (เช่น อว 78.03/1234)\n"
            "2. doc_date: วันที่ทำเอกสาร\n"
            "3. title: เรื่อง (เช่น ขออนุมัติจัดโครงการ...)\n"
            "4. requester: ผู้ทำการเบิก (ระบุชื่อบุคคล หรือชื่อภาควิชา/งาน)\n"
            "5. expense_items: รายละเอียดค่าใช้จ่าย (เบิกอะไรบ้าง และเท่าไหร่บ้าง) เป็น List ของ {item_no, description, quantity, unit, unit_price, total_price}\n"
            "6. total_amount: ยอดรวมเงินงบประมาณที่ขออนุมัติ (ตัวเลข Float)"
        ),
        "json_schema": {
            "doc_no": "เลขที่เอกสาร",
            "doc_date": "วันที่ทำเอกสาร",
            "title": "เรื่อง",
            "requester": "ผู้ทำการเบิก (บุคคล หรือ ภาควิชา)",
            "expense_items": [
                {"item_no": 1, "description": "ชื่อรายการค่าใช้จ่าย", "quantity": 1.0, "unit": "ชุด", "unit_price": 1000.0, "total_price": 1000.0}
            ],
            "total_amount": 1000.0
        }
    },
    DocumentType.PRINCIPLE_APPROVAL_GRANTED.value: {
        "title": "2. เอกสารอนุมัติหลักการ",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'เอกสารอนุมัติหลักการ'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. ref_doc_no: ตามหนังสือขออนุมัติหลักการเลขที่ (ถ้ามีระบุ)\n"
            "2. approval_items: รายละเอียดการอนุมัติ (อนุมัติรายการใดบ้าง และเท่าไหร่บ้าง)\n"
            "3. total_approved_amount: ยอดรวมเงินที่ได้รับการอนุมัติ (ตัวเลข Float)"
        ),
        "json_schema": {
            "ref_doc_no": "เลขที่หนังสือเดิมที่อ้างถึง",
            "approval_items": [
                {"item_no": 1, "description": "รายการที่อนุมัติ", "quantity": 1.0, "unit": "รายการ", "unit_price": 5000.0, "total_price": 5000.0}
            ],
            "total_approved_amount": 5000.0
        }
    },
    DocumentType.DISBURSEMENT_APPROVAL_REQUEST.value: {
        "title": "3. ขออนุมัติเบิกจ่าย",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของบันทึกข้อความ 'ขออนุมัติเบิกจ่าย'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร\n"
            "2. doc_date: วันที่ทำเอกสาร\n"
            "3. title: เรื่อง\n"
            "4. expense_items: รายละเอียดค่าใช้จ่าย (เบิกอะไรบ้าง และเท่าไหร่บ้าง)\n"
            "5. total_amount: ยอดรวม\n"
            "6. disbursement_type: ประเภทของการเบิกจ่าย (ระบุเป็น 'เงินสดย่อย' หรือ 'ทดรองจ่าย')"
        ),
        "json_schema": {
            "doc_no": "เลขที่เอกสาร",
            "doc_date": "วันที่ทำเอกสาร",
            "title": "เรื่อง",
            "expense_items": [
                {"item_no": 1, "description": "รายการเบิกจ่าย", "quantity": 1.0, "unit": "รายการ", "unit_price": 2500.0, "total_price": 2500.0}
            ],
            "total_amount": 2500.0,
            "disbursement_type": "เงินสดย่อย หรือ ทดรองจ่าย"
        }
    },
    DocumentType.ADVANCE_PAYMENT_REQUEST_1.value: {
        "title": "4. แบบเบิกเงินทดรองจ่าย (แบบที่ 1)",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'แบบเบิกเงินทดรองจ่าย (แบบที่ 1)'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร\n"
            "2. requester: ใครเป็นคนเบิก (ชื่อ-นามสกุล, ตำแหน่ง)\n"
            "3. total_amount: ยอดรวม\n"
            "4. ref_doc_no: ตามหนังสืออนุมัติเบิกจ่าย เลขที่อะไร\n"
            "5. expense_items: รายละเอียดตามประเภทค่าใช้จ่าย\n"
            "6. transfer_destination: โอนเงินไปที่ใด (ชื่อธนาคาร, เลขที่บัญชี หรือชื่อบัญชีผู้รับโอน)"
        ),
        "json_schema": {
            "doc_no": "เลขที่เอกสาร",
            "requester": "ชื่อผู้เบิก",
            "total_amount": 12000.0,
            "ref_doc_no": "ตามหนังสืออนุมัติเบิกจ่าย เลขที่...",
            "expense_items": [
                {"item_no": 1, "description": "ค่าใช้จ่ายย่อย", "quantity": 1.0, "unit": "งาน", "unit_price": 12000.0, "total_price": 12000.0}
            ],
            "transfer_destination": "ธนาคาร... เลขที่บัญชี..."
        }
    },
    DocumentType.ADVANCE_PAYMENT_REQUEST_2.value: {
        "title": "5. แบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์เงิน)",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'แบบเบิกเงินทดรองจ่าย (แบบที่ 2)'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร\n"
            "2. submission_date: วันที่ส่งเอกสาร\n"
            "3. claim_date: วันที่ขอรับเงิน\n"
            "4. requester: ใครเป็นคนเบิก\n"
            "5. total_amount: ยอดรวม\n"
            "6. ref_doc_no: ตามหนังสืออนุมัติหลักการ เลขที่อะไร\n"
            "7. expense_items: รายละเอียดค่าใช้จ่าย\n"
            "8. transfer_destination: โอนเงินไปที่ใด"
        ),
        "json_schema": {
            "doc_no": "เลขที่เอกสาร",
            "submission_date": "วันที่ส่งเอกสาร",
            "claim_date": "วันที่ขอรับเงิน",
            "requester": "ชื่อผู้เบิก",
            "total_amount": 8500.0,
            "ref_doc_no": "ตามหนังสืออนุมัติหลักการ เลขที่...",
            "expense_items": [
                {"item_no": 1, "description": "รายการ", "quantity": 1.0, "unit": "ครั้ง", "unit_price": 8500.0, "total_price": 8500.0}
            ],
            "transfer_destination": "บัญชีธนาคาร..."
        }
    },
    DocumentType.RECEIPT_SUBSTITUTE.value: {
        "title": "6. ใบแทนใบเสร็จ / ใบสำคัญรับเงิน",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'ใบแทนใบเสร็จ' หรือ 'ใบสำคัญรับเงิน'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_date: วัน/เดือน/ปี ที่จ่ายเงิน\n"
            "2. expense_items: รายละเอียดของรายการการเบิก (สินค้า/บริการ และยอดเงิน)\n"
            "3. total_amount: ยอดรวม\n"
            "4. payer: ผู้จ่ายเงิน (หรือผู้รับรองการจ่าย)"
        ),
        "json_schema": {
            "doc_date": "วัน/เดือน/ปี",
            "expense_items": [
                {"item_no": 1, "description": "ค่าจ้างเหมา...", "quantity": 1.0, "unit": "ครั้ง", "unit_price": 1500.0, "total_price": 1500.0}
            ],
            "total_amount": 1500.0,
            "payer": "ชื่อผู้จ่ายเงิน"
        }
    },
    DocumentType.PARCEL_INSPECTION.value: {
        "title": "7. ใบตรวจรับพัสดุ",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'ใบตรวจรับพัสดุ'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร (เช่น บพ. ...)\n"
            "2. inspectors: ผู้ตรวจรับพัสดุ (ระบุรายชื่อคณะกรรมการตรวจรับพัสดุ หรือผู้ตรวจรับ)\n"
            "3. inspection_date: วันที่ตรวจรับพัสดุ\n"
            "4. ref_doc_no: ตามใบสั่งซื้อ/สัญญาเลขที่ (ถ้ามี)"
        ),
        "json_schema": {
            "doc_no": "เลขที่เอกสาร",
            "inspectors": "ชื่อคณะกรรมการหรือผู้ตรวจรับพัสดุ",
            "inspection_date": "วันที่ตรวจรับ",
            "ref_doc_no": "เลขที่ใบสั่งซื้อหรือสัญญา"
        }
    },
    DocumentType.PROCUREMENT_APPROVAL_REQUEST.value: {
        "title": "8. ขออนุมัติจัดหาพัสดุ",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของบันทึก 'ขออนุมัติจัดหาพัสดุ'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร\n"
            "2. doc_date: วันที่ทำเอกสาร\n"
            "3. title: เรื่อง (เช่น ขออนุมัติจัดซื้อครุภัณฑ์...)\n"
            "4. procurement_reason: เหตุผลความจำเป็นที่ต้องจัดหา\n"
            "5. item_details: รายละเอียดของพัสดุ (รายการ, จำนวน, หน่วย, ราคาประมาณการ)\n"
            "6. total_budget: วงเงินงบประมาณที่ใช้ทั้งหมด (ตัวเลข Float)\n"
            "7. required_date: เวลาที่ต้องใช้พัสดุ (เช่น ภายใน 30 วัน, ภายในวันที่...)"
        ),
        "json_schema": {
            "doc_no": "เลขที่เอกสาร",
            "doc_date": "วันที่ทำเอกสาร",
            "title": "เรื่อง",
            "procurement_reason": "เหตุผลที่ต้องจัดหา",
            "item_details": [
                {"item_no": 1, "description": "ชื่อรายการพัสดุ", "quantity": 1.0, "unit": "เครื่อง", "unit_price": 35000.0, "total_price": 35000.0}
            ],
            "total_budget": 35000.0,
            "required_date": "เวลาที่ต้องใช้พัสดุ"
        }
    },
    DocumentType.PROCUREMENT_ATTACHMENT.value: {
        "title": "9. เอกสารประกอบการขออนุมัติจัดหา",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'เอกสารประกอบการขออนุมัติจัดหา'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. ref_memo_no: แนบท้ายบันทึกเอกสารเลขอะไร (เช่น แนบท้ายบันทึกข้อความ ที่ อว ...)\n"
            "2. expense_items: รายละเอียดพัสดุ/รายการตารางเปรียบเทียบราคา\n"
            "3. total_amount: ยอดรวม (ตัวเลข Float)"
        ),
        "json_schema": {
            "ref_memo_no": "แนบท้ายบันทึกเอกสารเลขที่...",
            "expense_items": [
                {"item_no": 1, "description": "รายการพัสดุ", "quantity": 1.0, "unit": "ชุด", "unit_price": 4500.0, "total_price": 4500.0}
            ],
            "total_amount": 4500.0
        }
    },
    DocumentType.GENERAL_RECEIPT.value: {
        "title": "10. ใบเสร็จรับเงิน / ใบกำกับภาษีทั่วไป",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'ใบเสร็จรับเงิน / ใบกำกับภาษี / บิลเงินสด / ใบสั่งซื้อ'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้ โดยยึดข้อมูลตามที่ปรากฏในเอกสาร 100%:\n"
            "1. vendor_name: ชื่อร้านค้า บริษัท หรือแบรนด์ผู้ขาย (ต้องเป็นชื่อเฉพาะของร้าน เช่น 'Pimploen\'s Shop' ห้ามตอบคำที่เป็นเพียงหัวข้อกำกับ เช่น 'ร้านค้าผู้ให้บริการ', 'ข้อมูลร้านค้า', 'ผู้ขาย')\n"
            "2. vendor_tax_id: เลขประจำตัวผู้เสียภาษี 13 หลักของผู้ขาย (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "3. vendor_branch: สาขา (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "4. vendor_address: ที่อยู่ร้านค้า (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "5. customer_name: ชื่อผู้ซื้อหรือชื่อลูกค้าที่พิมพ์ปรากฏบนเอกสาร (ต้องดึงชื่อบุคคลหรือลูกค้าจริง เช่น 'น้องพาเพลิน' ห้ามตอบคำที่เป็นเพียงหัวข้อกำกับ เช่น 'รายละเอียดลูกค้า', 'รายละเอียดลูกค้าคนสำคัญ', 'ข้อมูลผู้ซื้อ' หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "6. customer_tax_id: เลขประจำตัวผู้เสียภาษีของผู้ซื้อ (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "7. invoice_no: เลขที่ใบเสร็จ เลขที่บิล หรือเลขที่เอกสาร\n"
            "8. doc_date: วันที่บนเอกสาร\n"
            "9. line_items: รายการสินค้า/บริการ ทั้งหมด [item_no, description, quantity, unit, unit_price, total_price] รวมถึงค่าจัดส่งหรือส่วนลด (ถ้ามีระบุในบิล)\n"
            "10. subtotal: ยอดรวมราคาสินค้าก่อนหักส่วนลดหรือก่อนภาษี (เช่น 'ทั้งหมด', 'รวมเป็นเงิน') หากไม่มีให้ใส่ null\n"
            "11. vat: ยอดภาษีมูลค่าเพิ่ม (สกัดเฉพาะเมื่อเอกสารมีพิมพ์ระบุคำว่า 'ภาษีมูลค่าเพิ่ม' หรือ 'VAT' เท่านั้น หากไม่มีการกล่าวถึงภาษีในเอกสาร ให้ใส่เป็น null โดยเด็ดขาด ห้ามคำนวณ 7% เอง)\n"
            "12. total_amount: ยอดเงินรวมสุทธิที่ต้องชำระจริงตามที่ระบุบนเอกสาร (เช่น 'รวมราคาสุทธิ', 'ยอดชำระ', 'รวมทั้งสิ้น') ให้ดึงตัวเลขจากเอกสารโดยตรง ห้ามบวกเลขภาษีเพิ่มเอง"
        ),
        "json_schema": {
            "vendor_name": "ชื่อร้านค้าจริง (ห้ามตอบคำว่า ร้านค้าผู้ให้บริการ)",
            "vendor_tax_id": "ไม่มีระบุในเอกสาร",
            "vendor_branch": "ไม่มีระบุในเอกสาร",
            "vendor_address": "ไม่มีระบุในเอกสาร",
            "customer_name": "ชื่อลูกค้าจริง (ห้ามตอบคำว่า รายละเอียดลูกค้าคนสำคัญ)",
            "customer_tax_id": "ไม่มีระบุในเอกสาร",
            "invoice_no": "เลขที่เอกสาร",
            "doc_date": "วันที่",
            "line_items": [
                {"item_no": 1, "description": "ชื่อรายการสินค้าหรือบริการ", "quantity": 1.0, "unit": "ชิ้น", "unit_price": 490.0, "total_price": 490.0}
            ],
            "subtotal": 490.0,
            "vat": None,
            "total_amount": 570.0
        }
    }
}


def build_system_prompt_for_type(doc_type: str) -> str:
    """Construct a targeted system prompt for the specified document type."""
    config = PROMPT_CONFIGS.get(doc_type, PROMPT_CONFIGS[DocumentType.GENERAL_RECEIPT.value])
    schema_example = json.dumps(config["json_schema"], ensure_ascii=False, indent=2)

    return f"""คุณคือผู้เชี่ยวชาญด้านการสกัดข้อมูลเอกสารการเงินและเอกสารราชการ (Document Information Extraction AI)

ประเภทเอกสารเป้าหมาย: {config['title']}

{config['instructions']}

กฎเหล็กสำคัญอย่างยิ่งในการแยกแยะ 'หัวข้อกำกับ' (Label/Header) ออกจาก 'ชื่อจริง' (Actual Name):
1. **ห้ามนำคำที่เป็นเพียงหัวข้อกำกับมาตอบเป็นชื่อเด็ดขาด**:
   - คำทั่วไป เช่น 'ร้านค้าผู้ให้บริการ', 'ผู้ให้บริการ', 'ข้อมูลร้านค้า', 'ผู้ขาย', 'ชื่อร้าน' เป็นเพียงคำกำกับหัวข้อ (Section Header) **ไม่ใช่ชื่อร้านค้า!** ห้ามนำคำเหล่านี้มาใส่เป็น vendor_name เด็ดขาด ให้สกัดชื่อเฉพาะที่เป็นชื่อร้าน/แบรนด์/ธุรกิจ เช่น 'Pimploen\'s Shop'
   - คำทั่วไป เช่น 'รายละเอียดลูกค้า', 'รายละเอียดลูกค้าคนสำคัญ', 'ข้อมูลผู้ซื้อ', 'ลูกค้า', 'ผู้ซื้อ' เป็นเพียงคำกำกับหัวข้อ (Section Header) **ไม่ใช่ชื่อลูกค้า!** ห้ามนำคำเหล่านี้มาใส่เป็น customer_name เด็ดขาด ให้สกัดชื่อบุคคลที่เป็นชื่อลูกค้าจริง เช่น 'น้องพาเพลิน'
   - ในเอกสารที่มีการจัดรูปแบบ 2 คอลัมน์ (ซ้าย=ร้านค้า, ขวา=ลูกค้า):
     หัวข้อ 'ร้านค้าผู้ให้บริการ' คู่กับชื่อร้าน 'Pimploen\'s Shop'
     หัวข้อ 'รายละเอียดลูกค้าคนสำคัญ' คู่กับชื่อลูกค้า 'น้องพาเพลิน'
     ให้จับคู่ชื่อจริงใต้หัวข้อให้ถูกต้อง ห้ามสลับกันและห้ามนำหัวข้อมาตอบ

กฎเหล็กและแนวทางการสกัดข้อมูลทั่วไป (Strict Anti-Hallucination Rules):
2. **ยึดข้อเท็จจริงตามเอกสาร 100% (Zero Hallucination)**:
   - สกัดเฉพาะข้อมูลที่มีข้อความปรากฏชัดเจนในข้อความ OCR เท่านั้น
   - ห้ามเดา ห้ามคิดไปเอง ห้ามสมมุติ หรือจินตนาการข้อมูลขึ้นมาเองโดยเด็ดขาด เน้นความถูกต้อง ไม่เน้นความคิดสร้างสรรค์
3. **ข้อมูลที่ไม่มีระบุในเอกสาร**:
   - หากฟิลด์ใดไม่มีข้อความปรากฏในเอกสาร ให้ใส่เป็น "ไม่มีระบุในเอกสาร" หรือ null
   - โดยเฉพาะชื่อผู้ซื้อ (customer_name): ให้สกัดเฉพาะชื่อบุคคลหรือลูกค้าที่พิมพ์อยู่บนเอกสารเท่านั้น หากไม่มีระบุให้ใส่ "ไม่มีระบุในเอกสาร" ห้ามใส่ชื่อสถาบันหรือหน่วยงานใดๆ ที่ไม่มีพิมพ์ในเอกสารเด็ดขาด
4. **เรื่องภาษีมูลค่าเพิ่ม (VAT) ห้ามคิดคำนวณเอง**:
   - สกัดยอด VAT เฉพาะเมื่อในเอกสารมีการพิมพ์คำว่า "ภาษีมูลค่าเพิ่ม", "VAT", "ภาษี 7%" หรือมีตัวเลขภาษีระบุไว้ชัดเจนเท่านั้น
   - หากในเอกสาร **ไม่มีการระบุเรื่องภาษีมูลค่าเพิ่มเลย ให้ใส่ vat เป็น null โดยเด็ดขาด ห้ามคำนวณ 7% เองเด็ดขาด!**
5. **ยอดเงินรวมสุทธิ (total_amount)**:
   - ต้องสกัดจากตัวเลขยอดเงินสุทธิที่พิมพ์ระบุอยู่บนเอกสารจริง (เช่น 'รวมราคาสุทธิ', 'ยอดสุทธิ', 'ยอดชำระ', 'รวมทั้งสิ้น', 'Grand Total')
   - ห้ามนำราคาสินค้าไปบวกเลขภาษีเพิ่มเองจนตัวเลขไม่ตรงกับยอดสุทธิที่พิมพ์ในเอกสาร
6. **การคิดวิเคราะห์**:
   - หากโมเดลสามารถคิดได้ ให้เขียนขั้นตอนวิเคราะห์ในแท็ก <think>...</think> สั้นๆ ตรงไปตรงมา
7. **รูปแบบผลลัพธ์**:
   - ตอบผลลัพธ์เป็น JSON Object เท่านั้น โดยมีฟิลด์ตามโครงสร้างด้านล่างอย่างเคร่งครัด
   - ตัวเลขยอดเงินให้เป็น Float (เช่น 570.0) ห้ามใส่เครื่องหมายจุลภาคคั่น

โครงสร้าง JSON ที่ต้องการ:
{schema_example}
"""


class LLMExtractor:
    """
    Reasoning & Extraction Engine utilizing a local LLM via Ollama.
    Supports targeted extraction for 9 university reimbursement document types
    and commercial receipts, with fallback to authentic captured snapshots.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://127.0.0.1:11434",
        default_model: str = "qwen2.5:3b",
        timeout_seconds: int = 120
    ):
        self.base_url = ollama_base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout_seconds
        self.validator = FinancialDocumentValidator()

    def is_ollama_running(self) -> bool:
        """Check if local Ollama service is reachable on port 11434."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_available_models(self) -> List[str]:
        """Fetch list of models installed in local Ollama."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name") for m in data.get("models", [])]
        except Exception:
            return []

    def get_status(self) -> Dict[str, Any]:
        """Return system status for Ollama and fixture cache."""
        online = self.is_ollama_running()
        models = self.list_available_models() if online else []
        return {
            "ollama_online": online,
            "available_models": models,
            "default_model": self.default_model,
            "has_real_fixture": REAL_RESPONSE_FIXTURE.exists(),
            "fixture_path": str(REAL_RESPONSE_FIXTURE),
            "supported_document_types": [
                {"id": dt.value, "title": DOCUMENT_TYPE_TITLES[dt.value]}
                for dt in DocumentType
            ]
        }

    def extract(
        self,
        ocr_markdown: str,
        document_type: str = DocumentType.GENERAL_RECEIPT.value,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        use_mock_if_offline: bool = True,
        force_mock: bool = False
    ) -> TypeDirectedExtractionResult:
        """
        Extract structured data according to the targeted document type.
        """
        target_model = model_name or self.default_model
        start_time = time.time()

        if not force_mock and self.is_ollama_running():
            try:
                result = self._call_ollama_api(
                    ocr_markdown,
                    document_type,
                    target_model,
                    temperature=temperature,
                    start_time=start_time
                )
                return result
            except Exception as e:
                print(f"[LLMExtractor] Ollama call failed ({e}). Falling back to mock fixture.")
                if not use_mock_if_offline:
                    raise

        if use_mock_if_offline or force_mock:
            return self._generate_realistic_mock(ocr_markdown, document_type, target_model, start_time)

        raise ConnectionError("Ollama is not running and use_mock_if_offline is disabled.")

    def _call_ollama_api(
        self,
        ocr_markdown: str,
        document_type: str,
        model_name: str,
        temperature: float,
        start_time: float
    ) -> TypeDirectedExtractionResult:
        """Execute chat request against Ollama using the targeted document prompt."""
        system_prompt = build_system_prompt_for_type(document_type)
        type_title = DOCUMENT_TYPE_TITLES.get(document_type, document_type)
        user_prompt = (
            f"กรุณาวิเคราะห์และสกัดข้อมูลสำหรับ '{type_title}' จากเนื้อหาเอกสาร OCR ด้านล่างนี้ "
            f"โดยยึดข้อเท็จจริงตามเอกสาร 100% ห้ามคิดหรือคำนวณภาษีเพิ่มเองเด็ดขาด:\n\n{ocr_markdown}"
        )

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "top_p": 0.1,
                "num_gpu": 0
            }
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw_body = json.loads(resp.read().decode("utf-8"))

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        message_content = raw_body.get("message", {}).get("content", "")

        # Parse <think>...</think> if present
        thinking_process, clean_json_str = self._extract_thinking_and_json(message_content)
        parsed_data = self._parse_json_resilient(clean_json_str)

        return self._build_result_object(
            document_type=document_type,
            data=parsed_data,
            thinking_process=thinking_process,
            model_used=model_name,
            latency_ms=elapsed_ms,
            is_mock=False,
            raw_llm_response=message_content
        )

    def _build_result_object(
        self,
        document_type: str,
        data: Dict[str, Any],
        thinking_process: Optional[str],
        model_used: str,
        latency_ms: float,
        is_mock: bool,
        raw_llm_response: str
    ) -> TypeDirectedExtractionResult:
        """Normalizes extracted dictionary into TypeDirectedExtractionResult."""
        type_title = DOCUMENT_TYPE_TITLES.get(document_type, document_type)

        # Extract items list if present under various keys
        raw_items = (
            data.get("expense_items")
            or data.get("line_items")
            or data.get("approval_items")
            or data.get("item_details")
            or []
        )
        expense_items: List[ExpenseItem] = []
        for i, item in enumerate(raw_items):
            if isinstance(item, dict):
                expense_items.append(ExpenseItem(
                    item_no=item.get("item_no", i + 1),
                    description=item.get("description", str(item)),
                    quantity=item.get("quantity"),
                    unit=item.get("unit"),
                    unit_price=item.get("unit_price"),
                    total_price=item.get("total_price")
                ))

        # Extract primary total amount
        total_amount = (
            data.get("total_amount")
            or data.get("total_approved_amount")
            or data.get("total_budget")
        )
        if total_amount is not None:
            try:
                total_amount = float(total_amount)
            except Exception:
                pass

        subtotal = data.get("subtotal")
        vat = data.get("vat")

        res_obj = TypeDirectedExtractionResult(
            document_type=document_type,
            document_type_name_th=type_title,
            fields=data,
            expense_items=expense_items,
            line_items=expense_items,
            total_amount=total_amount,
            subtotal=subtotal,
            vat=vat,
            vendor_name=data.get("vendor_name"),
            vendor_tax_id=data.get("vendor_tax_id"),
            vendor_branch=data.get("vendor_branch"),
            vendor_address=data.get("vendor_address"),
            customer_name=data.get("customer_name") or data.get("requester"),
            invoice_no=data.get("invoice_no") or data.get("doc_no"),
            document_date=data.get("doc_date") or data.get("document_date"),
            thinking_process=thinking_process,
            model_used=model_used,
            latency_ms=latency_ms,
            is_mock=is_mock,
            raw_llm_response=raw_llm_response
        )

        if self.validator:
            self.validator.validate(res_obj)

        return res_obj

    @staticmethod
    def _extract_thinking_and_json(content: str) -> Tuple[Optional[str], str]:
        """Extract <think>...</think> reasoning trace from model output."""
        thinking = None
        think_match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Strip markdown code fence if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Look for outermost curly braces
            brace_match = re.search(r"(\{.*\})", content, flags=re.DOTALL)
            json_str = brace_match.group(1).strip() if brace_match else content

        return thinking, json_str

    @staticmethod
    def _parse_json_resilient(json_str: str) -> Dict[str, Any]:
        """Parse JSON with auto-repair for trailing commas or common LLM syntax slips."""
        try:
            return json.loads(json_str)
        except Exception:
            # Auto-strip trailing commas
            repaired = re.sub(r",\s*([\]}])", r"\1", json_str)
            try:
                return json.loads(repaired)
            except Exception as e:
                print(f"[LLMExtractor] Failed to parse JSON: {e}")
                return {}

    def _save_real_fixture(self, result: TypeDirectedExtractionResult, model_name: str):
        """Save authentic Ollama execution output as benchmark reference."""
        try:
            data = result.model_dump()
            data["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            data["saved_model"] = model_name
            with open(REAL_RESPONSE_FIXTURE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[LLMExtractor] Saved real response fixture to: {REAL_RESPONSE_FIXTURE}")
        except Exception as e:
            print(f"[LLMExtractor] Warning: Could not save fixture: {e}")

    def _generate_realistic_mock(
        self,
        ocr_markdown: str,
        document_type: str,
        model_name: str,
        start_time: float
    ) -> TypeDirectedExtractionResult:
        """
        Generate authentic mock output tailored to the specific document type.
        """
        elapsed_ms = round((time.time() - start_time) * 1000 + 380, 2)

        # 10. General Receipt - If fixture exists, use authentic Ollama snapshot
        if document_type == DocumentType.GENERAL_RECEIPT.value and REAL_RESPONSE_FIXTURE.exists():
            try:
                with open(REAL_RESPONSE_FIXTURE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                return self._build_result_object(
                    document_type=document_type,
                    data=saved.get("fields", saved),
                    thinking_process=saved.get("thinking_process") or "วิเคราะห์ใบเสร็จจาก snapshot จริง",
                    model_used=f"{model_name} (Mock from Real Ollama Snapshot)",
                    latency_ms=elapsed_ms,
                    is_mock=True,
                    raw_llm_response=saved.get("raw_llm_response", "")
                )
            except Exception:
                pass

        # Domain mocks for each of the 9 university document types
        mock_generators = {
            DocumentType.PRINCIPLE_APPROVAL_REQUEST.value: {
                "doc_no": "อว 78.03/ว.0425",
                "doc_date": "15 สิงหาคม 2567",
                "title": "ขออนุมัติหลักการจัดโครงการสัมมนาเชิงปฏิบัติการวิศวกรรมปัญญาประดิษฐ์",
                "requester": "ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์",
                "expense_items": [
                    {"item_no": 1, "description": "ค่าตอบแทนวิทยากรบรรยาย (6 ชั่วโมง)", "quantity": 6.0, "unit": "ชั่วโมง", "unit_price": 1000.0, "total_price": 6000.0},
                    {"item_no": 2, "description": "ค่าอาหารว่างและเครื่องดื่มสำหรับผู้เข้าร่วม 50 คน", "quantity": 50.0, "unit": "ชุด", "unit_price": 50.0, "total_price": 2500.0},
                    {"item_no": 3, "description": "ค่าวัสดุและเอกสารประกอบการสัมมนา", "quantity": 50.0, "unit": "ชุด", "unit_price": 70.0, "total_price": 3500.0}
                ],
                "total_amount": 12000.0
            },
            DocumentType.PRINCIPLE_APPROVAL_GRANTED.value: {
                "ref_doc_no": "อว 78.03/ว.0425 ลงวันที่ 15 สิงหาคม 2567",
                "approval_items": [
                    {"item_no": 1, "description": "อนุมัติค่าตอบแทนวิทยากร", "quantity": 6.0, "unit": "ชั่วโมง", "unit_price": 1000.0, "total_price": 6000.0},
                    {"item_no": 2, "description": "อนุมัติค่าอาหารว่างและเครื่องดื่ม", "quantity": 50.0, "unit": "ชุด", "unit_price": 50.0, "total_price": 2500.0},
                    {"item_no": 3, "description": "อนุมัติค่าวัสดุและเอกสารสัมมนา", "quantity": 50.0, "unit": "ชุด", "unit_price": 70.0, "total_price": 3500.0}
                ],
                "total_approved_amount": 12000.0
            },
            DocumentType.DISBURSEMENT_APPROVAL_REQUEST.value: {
                "doc_no": "อว 78.03/0892",
                "doc_date": "25 สิงหาคม 2567",
                "title": "ขออนุมัติเบิกจ่ายเงินงบประมาณค่าใช้จ่ายโครงการสัมมนาเชิงปฏิบัติการ",
                "expense_items": [
                    {"item_no": 1, "description": "เบิกจ่ายค่าตอบแทนวิทยากรและค่าอาหารว่าง", "quantity": 1.0, "unit": "งาน", "unit_price": 8500.0, "total_price": 8500.0}
                ],
                "total_amount": 8500.0,
                "disbursement_type": "ทดรองจ่าย"
            },
            DocumentType.ADVANCE_PAYMENT_REQUEST_1.value: {
                "doc_no": "ยง. 12/2567",
                "requester": "ผศ.ดร.สมชาย ใจดี (ภาควิชาวิศวกรรมคอมพิวเตอร์)",
                "total_amount": 12000.0,
                "ref_doc_no": "อว 78.03/0892",
                "expense_items": [
                    {"item_no": 1, "description": "เงินทดรองจ่ายเพื่อเป็นค่าใช้จ่ายในการจัดสัมมนา", "quantity": 1.0, "unit": "โครงการ", "unit_price": 12000.0, "total_price": 12000.0}
                ],
                "transfer_destination": "ธนาคารไทยพาณิชย์ เลขที่บัญชี 333-2-12345-6 นายสมชาย ใจดี"
            },
            DocumentType.ADVANCE_PAYMENT_REQUEST_2.value: {
                "doc_no": "ยง. 12/2567-ค",
                "submission_date": "30 สิงหาคม 2567",
                "claim_date": "31 สิงหาคม 2567",
                "requester": "ผศ.ดร.สมชาย ใจดี",
                "total_amount": 12000.0,
                "ref_doc_no": "อว 78.03/ว.0425",
                "expense_items": [
                    {"item_no": 1, "description": "เคลียร์เงินยืมทดรองจ่ายโครงการสัมมนา AI (ตามหลักฐานใบเสร็จ)", "quantity": 1.0, "unit": "ชุด", "unit_price": 12000.0, "total_price": 12000.0}
                ],
                "transfer_destination": "พร้อมเพย์ 081-999-xxxx"
            },
            DocumentType.RECEIPT_SUBSTITUTE.value: {
                "doc_date": "20 สิงหาคม 2567",
                "expense_items": [
                    {"item_no": 1, "description": "ค่าจ้างเหมาพาหนะรับส่งวิทยากร (รถแท็กซี่)", "quantity": 2.0, "unit": "เที่ยว", "unit_price": 350.0, "total_price": 700.0}
                ],
                "total_amount": 700.0,
                "payer": "นายวิศวกร มุ่งมั่น (ผู้สำรองจ่าย)"
            },
            DocumentType.PARCEL_INSPECTION.value: {
                "doc_no": "บพ. 88/2567",
                "inspectors": "รศ.ดร.สมบัติ ประธานกรรมการ, ผศ.ดร.วิชัย กรรมการ, นางสาวมณีรัตน์ กรรมการและเลขานุการ",
                "inspection_date": "18 กันยายน 2567",
                "ref_doc_no": "ใบสั่งซื้อเลขที่ PO-EG-2567-089"
            },
            DocumentType.PROCUREMENT_APPROVAL_REQUEST.value: {
                "doc_no": "อว 78.03/พสด.0112",
                "doc_date": "5 กันยายน 2567",
                "title": "ขออนุมัติจัดหาวัสดุและอุปกรณ์คอมพิวเตอร์สำหรับห้องปฏิบัติการ",
                "procurement_reason": "เนื่องจากอุปกรณ์เดิมชำรุดเสียหาย และจำเป็นต้องใช้ในการเรียนการสอนภาคเรียนที่ 1/2567",
                "item_details": [
                    {"item_no": 1, "description": "จอภาพมอนิเตอร์ 27 นิ้ว 4K", "quantity": 5.0, "unit": "จอ", "unit_price": 9500.0, "total_price": 47500.0},
                    {"item_no": 2, "description": "แป้นพิมพ์และเมาส์ไร้สาย", "quantity": 5.0, "unit": "ชุด", "unit_price": 1200.0, "total_price": 6000.0}
                ],
                "total_budget": 53500.0,
                "required_date": "ภายในวันที่ 30 กันยายน 2567"
            },
            DocumentType.PROCUREMENT_ATTACHMENT.value: {
                "ref_memo_no": "แนบท้ายบันทึกข้อความ ที่ อว 78.03/พสด.0112",
                "expense_items": [
                    {"item_no": 1, "description": "จอภาพมอนิเตอร์ 27 นิ้ว (ตามสเปก มหาวิทยาลัย)", "quantity": 5.0, "unit": "จอ", "unit_price": 9500.0, "total_price": 47500.0},
                    {"item_no": 2, "description": "แป้นพิมพ์และเมาส์", "quantity": 5.0, "unit": "ชุด", "unit_price": 1200.0, "total_price": 6000.0}
                ],
                "total_amount": 53500.0
            }
        }

        mock_data = mock_generators.get(document_type, mock_generators[DocumentType.PRINCIPLE_APPROVAL_REQUEST.value])
        type_name = DOCUMENT_TYPE_TITLES.get(document_type, document_type)

        mock_think = (
            f"1. สกัดข้อมูลตามประเภทเอกสารเป้าหมาย: {type_name}\n"
            f"2. สกัดฟิลด์สำคัญ: {list(mock_data.keys())}\n"
            f"3. ยอดรวมเงินที่คำนวณได้: {mock_data.get('total_amount') or mock_data.get('total_approved_amount') or mock_data.get('total_budget')} บาท"
        )

        return self._build_result_object(
            document_type=document_type,
            data=mock_data,
            thinking_process=mock_think,
            model_used=f"{model_name} (Targeted Mock Mode)",
            latency_ms=elapsed_ms,
            is_mock=True,
            raw_llm_response=json.dumps(mock_data, ensure_ascii=False, indent=2)
        )
