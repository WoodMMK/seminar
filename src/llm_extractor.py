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
            "1. doc_no: เลขที่เอกสาร (เช่น อว 78.03/1234 หรือ ไม่มีระบุในเอกสาร)\n"
            "2. doc_date: วันที่ทำเอกสาร (หรือ ไม่มีระบุในเอกสาร)\n"
            "3. title: เรื่อง (เช่น ขออนุมัติจัดโครงการ... หรือ ไม่มีระบุในเอกสาร)\n"
            "4. requester: ผู้ขออนุมัติหลักการ (ระบุชื่อบุคคล หรือชื่อภาควิชา/หน่วยงาน หรือ ไม่มีระบุในเอกสาร)\n"
            "5. expense_items: รายละเอียดค่าใช้จ่าย (เบิกอะไรบ้าง และเท่าไหร่บ้าง) เป็น List ของ {item_no, description, quantity, unit, unit_price, total_price}\n"
            "6. total_amount: ยอดรวมเงินงบประมาณที่ขออนุมัติ (ตัวเลข Float หรือ null หากไม่มีระบุ)"
        ),
        "json_schema": {
            "doc_no": "<เลขที่เอกสารจริงจากข้อความ OCR หรือ ไม่มีระบุในเอกสาร>",
            "doc_date": "<วันที่ทำเอกสารจากข้อความ OCR หรือ ไม่มีระบุในเอกสาร>",
            "title": "<เรื่อง หรือ ไม่มีระบุในเอกสาร>",
            "requester": "<ผู้ขออนุมัติหลักการ หรือ ไม่มีระบุในเอกสาร>",
            "expense_items": [
                {"item_no": 1, "description": "<รายการค่าใช้จ่าย>", "quantity": 1.0, "unit": "<หน่วยนับ>", "unit_price": 0.0, "total_price": 0.0}
            ],
            "total_amount": 0.0
        }
    },
    DocumentType.PRINCIPLE_APPROVAL_GRANTED.value: {
        "title": "2. เอกสารอนุมัติหลักการ",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'เอกสารอนุมัติหลักการ'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. ref_doc_no: ตามหนังสือขออนุมัติหลักการเลขที่ (ถ้ามีระบุ หรือ ไม่มีระบุในเอกสาร)\n"
            "2. approval_items: รายละเอียดการอนุมัติ (อนุมัติรายการใดบ้าง และเท่าไหร่บ้าง)\n"
            "3. total_approved_amount: ยอดรวมเงินที่ได้รับการอนุมัติ (ตัวเลข Float หรือ null หากไม่มีระบุ)"
        ),
        "json_schema": {
            "ref_doc_no": "<เลขที่หนังสือเดิมที่อ้างถึง หรือ ไม่มีระบุในเอกสาร>",
            "approval_items": [
                {"item_no": 1, "description": "<รายการที่อนุมัติ>", "quantity": 1.0, "unit": "<หน่วยนับ>", "unit_price": 0.0, "total_price": 0.0}
            ],
            "total_approved_amount": 0.0
        }
    },
    DocumentType.DISBURSEMENT_APPROVAL_REQUEST.value: {
        "title": "3. ขออนุมัติเบิกจ่าย",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของบันทึกข้อความ 'ขออนุมัติเบิกจ่าย'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร (เช่น ที่ อว 78.101/20290 หรือ ไม่มีระบุในเอกสาร)\n"
            "2. doc_date: วันที่ทำเอกสาร (หรือ ไม่มีระบุในเอกสาร)\n"
            "3. title: เรื่อง (หรือ ไม่มีระบุในเอกสาร)\n"
            "4. expense_items: รายละเอียดค่าใช้จ่าย (เบิกอะไรบ้าง และเท่าไหร่บ้าง)\n"
            "5. total_amount: ยอดรวมเงินที่ขออนุมัติเบิกจ่าย (ตัวเลข Float หรือ null หากไม่มีระบุ)\n"
            "6. disbursement_type: ประเภทของการเบิกจ่าย (ระบุเป็น 'เงินสดย่อย' หรือ 'ทดรองจ่าย' หรือ ไม่มีระบุในเอกสาร)"
        ),
        "json_schema": {
            "doc_no": "<เลขที่เอกสารจริงจากข้อความ OCR หรือ ไม่มีระบุในเอกสาร>",
            "doc_date": "<วันที่ทำเอกสารจากข้อความ OCR หรือ ไม่มีระบุในเอกสาร>",
            "title": "<เรื่อง หรือ ไม่มีระบุในเอกสาร>",
            "expense_items": [
                {"item_no": 1, "description": "<รายการเบิกจ่าย>", "quantity": 1.0, "unit": "<หน่วยนับ>", "unit_price": 0.0, "total_price": 0.0}
            ],
            "total_amount": 0.0,
            "disbursement_type": "<เงินสดย่อย หรือ ทดรองจ่าย หรือ ไม่มีระบุในเอกสาร>"
        }
    },
    DocumentType.ADVANCE_PAYMENT_REQUEST_1.value: {
        "title": "4. แบบเบิกเงินทดรองจ่าย (แบบที่ 1 - สัญญายืมเงิน/เบิก)",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'แบบเบิกเงินทดรองจ่าย (แบบที่ 1)'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร (เช่น จปม เลขที่ B262/2568 หรือ ไม่มีระบุในเอกสาร)\n"
            "2. requester: ใครเป็นคนเบิก (ชื่อ-นามสกุล, ตำแหน่ง ของผู้ขอเบิกเงิน ห้ามตอบชื่อหัวหน้าภาควิชาหรือผู้ตรวจรับ)\n"
            "3. total_amount: ยอดรวมเงินทดรองจ่ายที่ขอเบิก (ตัวเลข Float เช่น 2100.0 ห้ามตอบ null หากมีตัวเลขในเอกสาร)\n"
            "4. ref_doc_no: ตามหนังสืออนุมัติเบิกจ่าย เลขที่อะไร (เช่น อว 78.101/20290)\n"
            "5. expense_items: รายละเอียดตามประเภทค่าใช้จ่าย\n"
            "6. transfer_destination: โอนเงินไปที่ใด (ชื่อธนาคาร, เลขที่บัญชี, ชื่อบัญชีผู้รับโอน)"
        ),
        "json_schema": {
            "doc_no": "<เลขที่เอกสารจริงจากข้อความ OCR เช่น B262/2568 หรือ ไม่มีระบุในเอกสาร>",
            "requester": "<ชื่อ-นามสกุลของผู้ขอเบิกเงินจริงจากเอกสาร หรือ ไม่มีระบุในเอกสาร>",
            "total_amount": 0.0,
            "ref_doc_no": "<เลขที่หนังสืออนุมัติที่อ้างอิงถึง หรือ ไม่มีระบุในเอกสาร>",
            "expense_items": [
                {"item_no": 1, "description": "<รายละเอียดค่าใช้จ่าย>", "quantity": 1.0, "unit": "<หน่วยนับ>", "unit_price": 0.0, "total_price": 0.0}
            ],
            "transfer_destination": "<ธนาคาร เลขที่บัญชี ชื่อบัญชี หรือ ไม่มีระบุในเอกสาร>"
        }
    },
    DocumentType.ADVANCE_PAYMENT_REQUEST_2.value: {
        "title": "5. แบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์เงิน)",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'แบบเบิกเงินทดรองจ่าย (แบบที่ 2)'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร\n"
            "2. submission_date: วันที่ส่งเอกสาร (หรือ ไม่มีระบุในเอกสาร)\n"
            "3. claim_date: วันที่ขอรับเงิน (หรือ ไม่มีระบุในเอกสาร)\n"
            "4. requester: ใครเป็นคนเบิก (ชื่อผู้ขอรับเงิน)\n"
            "5. total_amount: ยอดรวมเงิน (ตัวเลข Float)\n"
            "6. ref_doc_no: ตามหนังสืออนุมัติหลักการ เลขที่อะไร (หรือ ไม่มีระบุในเอกสาร)\n"
            "7. expense_items: รายละเอียดค่าใช้จ่าย\n"
            "8. transfer_destination: โอนเงินไปที่ใด (หรือ ไม่มีระบุในเอกสาร)"
        ),
        "json_schema": {
            "doc_no": "<เลขที่เอกสารจริงจากข้อความ OCR หรือ ไม่มีระบุในเอกสาร>",
            "submission_date": "<วันที่ส่งเอกสาร หรือ ไม่มีระบุในเอกสาร>",
            "claim_date": "<วันที่ขอรับเงิน หรือ ไม่มีระบุในเอกสาร>",
            "requester": "<ชื่อผู้เบิกจริงจากเอกสาร หรือ ไม่มีระบุในเอกสาร>",
            "total_amount": 0.0,
            "ref_doc_no": "<เลขที่หนังสืออ้างอิง หรือ ไม่มีระบุในเอกสาร>",
            "expense_items": [
                {"item_no": 1, "description": "<รายการค่าใช้จ่าย>", "quantity": 1.0, "unit": "<หน่วยนับ>", "unit_price": 0.0, "total_price": 0.0}
            ],
            "transfer_destination": "<บัญชีธนาคาร หรือ ไม่มีระบุในเอกสาร>"
        }
    },
    DocumentType.RECEIPT_SUBSTITUTE.value: {
        "title": "6. ใบแทนใบเสร็จ / ใบสำคัญรับเงิน",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'ใบแทนใบเสร็จ' หรือ 'ใบสำคัญรับเงิน'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_date: วัน/เดือน/ปี ที่จ่ายเงิน (หรือ ไม่มีระบุในเอกสาร)\n"
            "2. expense_items: รายละเอียดของรายการการเบิก (สินค้า/บริการ และยอดเงิน)\n"
            "3. total_amount: ยอดรวม (ตัวเลข Float)\n"
            "4. payer: ผู้จ่ายเงิน (หรือผู้รับรองการจ่าย หรือ ไม่มีระบุในเอกสาร)"
        ),
        "json_schema": {
            "doc_date": "<วัน/เดือน/ปี หรือ ไม่มีระบุในเอกสาร>",
            "expense_items": [
                {"item_no": 1, "description": "<รายการค่าใช้จ่าย>", "quantity": 1.0, "unit": "<หน่วยนับ>", "unit_price": 0.0, "total_price": 0.0}
            ],
            "total_amount": 0.0,
            "payer": "<ชื่อผู้จ่ายเงิน หรือ ไม่มีระบุในเอกสาร>"
        }
    },
    DocumentType.PARCEL_INSPECTION.value: {
        "title": "7. ใบตรวจรับพัสดุ",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'ใบตรวจรับพัสดุ'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร (เช่น บพ. ... หรือ ไม่มีระบุในเอกสาร)\n"
            "2. inspectors: ผู้ตรวจรับพัสดุ (ระบุรายชื่อคณะกรรมการตรวจรับพัสดุ หรือผู้ตรวจรับ)\n"
            "3. inspection_date: วันที่ตรวจรับพัสดุ (หรือ ไม่มีระบุในเอกสาร)\n"
            "4. ref_doc_no: ตามใบสั่งซื้อ/สัญญาเลขที่ (ถ้ามี หรือ ไม่มีระบุในเอกสาร)"
        ),
        "json_schema": {
            "doc_no": "<เลขที่เอกสารจริง หรือ ไม่มีระบุในเอกสาร>",
            "inspectors": "<ชื่อคณะกรรมการหรือผู้ตรวจรับพัสดุ หรือ ไม่มีระบุในเอกสาร>",
            "inspection_date": "<วันที่ตรวจรับ หรือ ไม่มีระบุในเอกสาร>",
            "ref_doc_no": "<เลขที่ใบสั่งซื้อหรือสัญญา หรือ ไม่มีระบุในเอกสาร>"
        }
    },
    DocumentType.PROCUREMENT_APPROVAL_REQUEST.value: {
        "title": "8. ขออนุมัติจัดหาพัสดุ",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของบันทึก 'ขออนุมัติจัดหาพัสดุ'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. doc_no: เลขที่เอกสาร (หรือ ไม่มีระบุในเอกสาร)\n"
            "2. doc_date: วันที่ทำเอกสาร (หรือ ไม่มีระบุในเอกสาร)\n"
            "3. title: เรื่อง (หรือ ไม่มีระบุในเอกสาร)\n"
            "4. procurement_reason: เหตุผลความจำเป็นที่ต้องจัดหา (หรือ ไม่มีระบุในเอกสาร)\n"
            "5. item_details: รายละเอียดของพัสดุ (รายการ, จำนวน, หน่วย, ราคาประมาณการ)\n"
            "6. total_budget: วงเงินงบประมาณที่ใช้ทั้งหมด (ตัวเลข Float)\n"
            "7. required_date: เวลาที่ต้องใช้พัสดุ (หรือ ไม่มีระบุในเอกสาร)"
        ),
        "json_schema": {
            "doc_no": "<เลขที่เอกสารจริง หรือ ไม่มีระบุในเอกสาร>",
            "doc_date": "<วันที่ทำเอกสาร หรือ ไม่มีระบุในเอกสาร>",
            "title": "<เรื่อง หรือ ไม่มีระบุในเอกสาร>",
            "procurement_reason": "<เหตุผลที่ต้องจัดหา หรือ ไม่มีระบุในเอกสาร>",
            "item_details": [
                {"item_no": 1, "description": "<ชื่อรายการพัสดุ>", "quantity": 1.0, "unit": "<หน่วยนับ>", "unit_price": 0.0, "total_price": 0.0}
            ],
            "total_budget": 0.0,
            "required_date": "<เวลาที่ต้องใช้พัสดุ หรือ ไม่มีระบุในเอกสาร>"
        }
    },
    DocumentType.PROCUREMENT_ATTACHMENT.value: {
        "title": "9. เอกสารประกอบการขออนุมัติจัดหา",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'เอกสารประกอบการขออนุมัติจัดหา'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้:\n"
            "1. ref_memo_no: แนบท้ายบันทึกเอกสารเลขอะไร (เช่น แนบท้ายบันทึกข้อความ ที่ อว ... หรือ ไม่มีระบุในเอกสาร)\n"
            "2. expense_items: รายละเอียดพัสดุ/รายการตารางเปรียบเทียบราคา\n"
            "3. total_amount: ยอดรวม (ตัวเลข Float หรือ null หากไม่มีระบุ)"
        ),
        "json_schema": {
            "ref_memo_no": "<แนบท้ายบันทึกเอกสารเลขที่... หรือ ไม่มีระบุในเอกสาร>",
            "expense_items": [
                {"item_no": 1, "description": "<รายการพัสดุ>", "quantity": 1.0, "unit": "<หน่วยนับ>", "unit_price": 0.0, "total_price": 0.0}
            ],
            "total_amount": 0.0
        }
    },
    DocumentType.GENERAL_RECEIPT.value: {
        "title": "10. ใบเสร็จรับเงิน / ใบกำกับภาษีทั่วไป",
        "instructions": (
            "คุณกำลังวิเคราะห์เนื้อหา OCR ของ 'ใบเสร็จรับเงิน / ใบกำกับภาษี / บิลเงินสด / ใบสั่งซื้อ'\n"
            "กรุณาสกัดข้อมูลเฉพาะเจาะจงดังต่อไปนี้ โดยยึดข้อมูลตามที่ปรากฏในเอกสาร 100%:\n"
            "1. vendor_name: ชื่อร้านค้า บริษัท หรือแบรนด์ผู้ขายจริง (ห้ามตอบคำที่เป็นเพียงหัวข้อกำกับ เช่น 'ร้านค้าผู้ให้บริการ', 'ผู้ขาย')\n"
            "2. vendor_tax_id: เลขประจำตัวผู้เสียภาษี 13 หลักของผู้ขาย (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "3. vendor_branch: สาขา (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "4. vendor_address: ที่อยู่ร้านค้า (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "5. customer_name: ชื่อผู้ซื้อหรือชื่อลูกค้าจริง (ห้ามตอบคำที่เป็นเพียงหัวข้อกำกับ เช่น 'รายละเอียดลูกค้าคนสำคัญ')\n"
            "6. customer_tax_id: เลขประจำตัวผู้เสียภาษีของผู้ซื้อ (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "7. invoice_no: เลขที่ใบเสร็จ เลขที่บิล หรือเลขที่เอกสาร (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "8. doc_date: วันที่บนเอกสาร (หากไม่มีให้ใส่ 'ไม่มีระบุในเอกสาร')\n"
            "9. line_items: รายการสินค้า/บริการ ทั้งหมด [item_no, description, quantity, unit, unit_price, total_price]\n"
            "10. subtotal: ยอดรวมราคาสินค้าก่อนหักส่วนลดหรือก่อนภาษี (ตัวเลข float หรือ null หากไม่มีระบุ)\n"
            "11. vat: ยอดภาษีมูลค่าเพิ่ม (สกัดเฉพาะเมื่อเอกสารมีพิมพ์ระบุคำว่า 'ภาษีมูลค่าเพิ่ม' หรือ 'VAT' เท่านั้น หากไม่มีการกล่าวถึงภาษีในเอกสาร ให้ใส่เป็น null โดยเด็ดขาด ห้ามคำนวณ 7% เอง)\n"
            "12. total_amount: ยอดเงินรวมสุทธิที่ต้องชำระจริงตามที่ระบุบนเอกสาร (ตัวเลข float หรือ null หากไม่มีระบุ)"
        ),
        "json_schema": {
            "vendor_name": "<ชื่อร้านค้าจริง หรือ ไม่มีระบุในเอกสาร>",
            "vendor_tax_id": "ไม่มีระบุในเอกสาร",
            "vendor_branch": "ไม่มีระบุในเอกสาร",
            "vendor_address": "ไม่มีระบุในเอกสาร",
            "customer_name": "<ชื่อลูกค้าจริง หรือ ไม่มีระบุในเอกสาร>",
            "customer_tax_id": "ไม่มีระบุในเอกสาร",
            "invoice_no": "<เลขที่เอกสาร หรือ ไม่มีระบุในเอกสาร>",
            "doc_date": "<วันที่ หรือ ไม่มีระบุในเอกสาร>",
            "line_items": [
                {"item_no": 1, "description": "<ชื่อรายการสินค้าหรือบริการ>", "quantity": 1.0, "unit": "<หน่วยนับ>", "unit_price": 0.0, "total_price": 0.0}
            ],
            "subtotal": 0.0,
            "vat": None,
            "total_amount": 0.0
        }
    }
}


DATA_EXTRACTOR_BASE_TEMPLATE = """# SYSTEM PROMPT: Enterprise-Grade Financial & Official Document Data Extractor

## PART 1: CORE OPERATING DIRECTIVES & GENERAL DATA EXTRACTION STANDARDS
You are an expert, deterministic Information Extraction Engine specialized in Thai official government memos, university financial workflows, reimbursement forms, and tax receipts.
Your objective is to extract structured JSON data from OCR-transcribed document text with maximum precision, strict zero-hallucination compliance, and robust tolerance to optical scanning imperfections.

### 1. STRICT ZERO-HALLUCINATION & FACTUAL FIDELITY MANDATE
- **Ground-Truth Bound**: Extract ONLY facts, names, figures, and dates that explicitly exist in the OCR text.
- **Forbidden Hallucinations**: Never fabricate missing values. Never output prompt template example names (e.g., 'สมชาย', 'บริษัท ตัวอย่าง จำกัด') or arbitrary sample numbers.
- **Deterministic Missing Value Policy**:
  - Missing text fields: `"ไม่มีระบุในเอกสาร"`
  - Missing numeric/financial fields: `null` (never invent 0.0 unless the text explicitly states 0.0 or ฟรี)
  - Missing date fields: `"ไม่มีระบุในเอกสาร"` or `null`
  - Missing list/array fields: `[]` (empty list)
  - Never guess, invent, or extrapolate.

### 2. OCR ROBUSTNESS, THAI PHONETIC INTERPRETATION & CANONICAL SPELLING
- **LLM Semantic Disambiguation**: Raw OCR text inevitably contains missing vowels, dropped tone marks, or merged words due to paper scanning conditions (e.g. 'เบกเงน' means 'เบิกเงิน', 'ทดรองจาย' means 'ทดรองจ่าย', 'ชาระเงน' means 'ชำระเงิน', 'เจาหนาที่บรหารงาน' means 'เจ้าหน้าที่บริหารงาน', 'ตอนรบคณะ' means 'ต้อนรับคณะ', 'คณบดีี' means 'คณบดี', 'ถปม เลขที่' means 'จปม. เลขที่'). You MUST use your semantic understanding of Thai language, government terminology, and contextual clues to output clean, correctly spelled canonical Thai text.
- **Names & Titles**: Normalize official titles and personal names to standard Thai orthography (e.g. 'นางสาว', 'นาย', 'ผศ.ดร.', 'รศ.ดร.'). Clean minor OCR typos in recognized names based on surrounding context.
- **Document Numbers**: Extract the clean identifier without stray prefixes or OCR artifacts (e.g. from 'จปม เลขที่ B262/2568' extract 'B262/2568'; from 'ที่ อว 78.101/334' extract 'อว 78.101/334').
- **Label vs Value Disambiguation**: Do NOT capture descriptive labels (e.g., 'ข้าพเจ้า', 'ผู้ขอเบิก', 'ชื่อผู้รับเงิน', 'ผู้ขาย', 'ร้านค้าผู้ให้บริการ', 'รายละเอียดลูกค้า') as the actual names. Extract the entity that appears immediately following the label.

### 3. THAI GOVERNMENT REIMBURSEMENT ENTITY RULES
- **Primary Document ID (`doc_no`) vs Referenced ID (`ref_doc_no` / `ref_memo_no`)**:
  - `doc_no`: The document's own identifier (e.g. 'เลขที่ B262/2568', 'จปม. เลขที่...', 'ที่ อว 78.101/...').
  - `ref_doc_no`: Any prior letter or approval being referenced or cited (e.g., 'ตามหนังสืออนุมัติเบิกจ่าย เลขที่...', 'อ้างถึง...').
- **Requester (`requester`) vs Authorizing Signatures**:
  - `requester`: The person filing the claim / requesting funds (usually introduced by 'ข้าพเจ้า...', 'ผู้ขอเบิก...').
  - Do NOT confuse the requester with department heads, deans, treasurers, committee members, or inspectors signing approval sections at the bottom (e.g. 'หัวหน้าภาควิชา', 'คณบดี', 'ประธานกรรมการ').
- **Currency & Amount Cross-Verification**:
  - When the document provides both numerical figures and Thai textual currency in parentheses (e.g. '210000 บาท (สองพันหนึ่งร้อยบาทถ้วน)' or '2,100.00 บาท'), cross-verify with the spelled-out Thai words. 'สองพันหนึ่งร้อยบาท' verifies that the true amount is 2,100.00, not 210,000!
  - Numeric fields must be extracted as clean numbers (Float, e.g. 2100.0), without commas or currency suffixes.
- **Tax & VAT Rule**:
  - Only record VAT when explicit words such as 'ภาษีมูลค่าเพิ่ม' or 'VAT' appear with an associated amount. Do NOT calculate 7% manually. If not stated, return `null`.

### 4. DATE FORMATTING
- Maintain Thai calendar years (BE, e.g. 2568, 2569) as printed on official documents. Remove accidental OCR punctuation inside dates (e.g. '14.สิงหาคม.2568' -> '14 สิงหาคม 2568').
"""


def build_system_prompt_for_type(doc_type: str) -> str:
    """
    Constructs a two-stage composite system prompt:
    1. Base Data Extractor Template (enterprise directives, zero-hallucination, OCR robustness, Thai government entity rules)
    2. Document-Specific Directives & JSON Schema for the target document type.
    """
    config = PROMPT_CONFIGS.get(doc_type, PROMPT_CONFIGS[DocumentType.GENERAL_RECEIPT.value])
    schema_example = json.dumps(config["json_schema"], ensure_ascii=False, indent=2)

    return f"""{DATA_EXTRACTOR_BASE_TEMPLATE}

## PART 2: DOCUMENT-SPECIFIC EXTRACTION INSTRUCTIONS
**Target Document**: {config['title']}

{config['instructions']}

## PART 3: OUTPUT JSON SCHEMA
Respond ONLY with a valid JSON object strictly matching this structure:
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
        timeout_seconds: int = 120,
        num_gpu: Optional[int] = None
    ):
        self.base_url = ollama_base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout_seconds
        # -1 instructs Ollama to offload 100% of layers to GPU if available (gracefully falls back to CPU)
        self.num_gpu = int(os.getenv("OLLAMA_NUM_GPU", "-1")) if num_gpu is None else num_gpu
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
            "gpu_acceleration": "auto_enabled (offload all layers if GPU present)" if self.num_gpu != 0 else "disabled",
            "num_gpu": self.num_gpu,
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
        **kwargs: Any,
    ) -> TypeDirectedExtractionResult:
        """
        Extract structured data according to the targeted document type.
        Raises ConnectionError if Ollama is not running, or re-raises any LLM call failure.
        """
        if not model_name or str(model_name).strip().lower() in ("string", "default", "none", ""):
            target_model = self.default_model
        else:
            target_model = model_name
        start_time = time.time()

        if not self.is_ollama_running():
            raise ConnectionError(
                f"Ollama is not running at {self.base_url}. "
                "Please start Ollama with 'ollama serve' before calling the extraction API."
            )

        return self._call_ollama_api(
            ocr_markdown,
            document_type,
            target_model,
            temperature=temperature,
            start_time=start_time
        )

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
            "format": "json",
            "options": {
                "temperature": float(temperature),
                "top_p": 0.1,
                "num_ctx": 8192,
                "num_gpu": self.num_gpu,  # -1 = auto-offload 100% of layers to GPU if available; fallback to CPU gracefully
            }
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        raw_body = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if attempt == 0 and e.code >= 500:
                    time.sleep(1.0)
                    if "format" in payload:
                        del payload["format"]
                        req = urllib.request.Request(
                            f"{self.base_url}/api/chat",
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST"
                        )
                    continue
                raise

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
                if "total_amount" not in data or data["total_amount"] is None:
                    data["total_amount"] = total_amount
                if document_type == DocumentType.PROCUREMENT_APPROVAL_REQUEST.value and ("total_budget" not in data or data["total_budget"] is None):
                    data["total_budget"] = total_amount
                if document_type == DocumentType.PRINCIPLE_APPROVAL_GRANTED.value and ("total_approved_amount" not in data or data["total_approved_amount"] is None):
                    data["total_approved_amount"] = total_amount
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
        if not json_str or not json_str.strip():
            return {}
        try:
            return json.loads(json_str)
        except Exception:
            # Auto-strip trailing commas
            repaired = re.sub(r",\s*([\]}])", r"\1", json_str)
            try:
                return json.loads(repaired)
            except Exception:
                sub_match = re.search(r"(\{.*\})", json_str, flags=re.DOTALL)
                if sub_match:
                    try:
                        return json.loads(sub_match.group(1))
                    except Exception:
                        pass
                print(f"[LLMExtractor] Failed to parse JSON: {json_str[:120]}...")
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

        # Dynamic OCR extraction: Extract real fields from ocr_markdown directly
        # Strictly enforces ZERO-HALLUCINATION: missing values are 'ไม่มีระบุในเอกสาร' or None
        doc_no_match = re.search(r'(?:จปม\s*เลขที่|เลขที่|ที่\s*อว|ที่|บพ\.)[\s\.:]*([A-Za-z0-9\./\-]+)', ocr_markdown)
        doc_no = doc_no_match.group(1).strip() if doc_no_match else "ไม่มีระบุในเอกสาร"

        date_match = re.search(r'(?:วันที่|เมื่อวันที่|ลงวันที่)[\s\.:]*([0-9]{1,2}\s+[^\s0-9]+\s+[0-9]{4})', ocr_markdown)
        doc_date = date_match.group(1).strip() if date_match else "ไม่มีระบุในเอกสาร"

        title_match = re.search(r'(?:เรื่อง|หัวข้อ)[\s\.:]*([^\n\r]+)', ocr_markdown)
        title = title_match.group(1).strip() if title_match else "ไม่มีระบุในเอกสาร"

        req_match = re.search(r'(?:ข้าพเจ้า|ผู้ขอเบิก|ขอรับรอง|โดยมี)\s+((?:นาย|นาง|นางสาว|ผศ\.|ดร\.|รศ\.|ศ\.)[^\s,]+(?:\s+[^\s,]+)?)', ocr_markdown)
        requester = req_match.group(1).strip() if req_match else "ไม่มีระบุในเอกสาร"

        amt_match = re.search(r'(?:เป็นเงิน|จำนวนเงิน|รวมทั้งสิ้น|รวมเป็นเงิน|ยอดรวม|จำนวน)[\s\.:]*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)', ocr_markdown)
        total_amount = None
        if amt_match:
            try:
                total_amount = float(amt_match.group(1).replace(",", ""))
            except Exception:
                total_amount = None

        ref_match = re.search(r'(?:ตามหนังสืออนุมัติเบิกจ่าย|ตามหนังสืออนุมัติ|ตามหนังสือ|อ้างถึง|อ้างอิง|เลขที่)\s*([A-Za-z0-9\./\-]+)', ocr_markdown)
        ref_doc_no = ref_match.group(1).strip() if ref_match else "ไม่มีระบุในเอกสาร"

        trans_match = re.search(r'(?:โอนเข้าบัญชี|ธนาคาร|เลขที่บัญชี)[\s\.:]*([^\n\r]+)', ocr_markdown)
        transfer_dest = trans_match.group(1).strip() if trans_match else "ไม่มีระบุในเอกสาร"

        item_match = re.search(r'(?:โดยมีรายละเอียดค่าใช้จ่าย|รายละเอียดค่าใช้จ่าย ดังนี้)[\s\.:]*\n*([^\n\r]+)', ocr_markdown)
        item_desc = item_match.group(1).strip() if item_match else (title if title != "ไม่มีระบุในเอกสาร" else "ค่าใช้จ่ายตามเอกสาร")

        expense_items = []
        if total_amount is not None:
            expense_items = [
                {
                    "item_no": 1,
                    "description": item_desc,
                    "quantity": 1.0,
                    "unit": "รายการ",
                    "unit_price": total_amount,
                    "total_price": total_amount
                }
            ]

        config = PROMPT_CONFIGS.get(document_type, PROMPT_CONFIGS[DocumentType.GENERAL_RECEIPT.value])
        schema_keys = set(config["json_schema"].keys())

        mock_data: Dict[str, Any] = {}
        for k in schema_keys:
            if k == "doc_no":
                mock_data[k] = doc_no
            elif k in ("doc_date", "submission_date", "claim_date", "inspection_date", "required_date"):
                mock_data[k] = doc_date
            elif k == "title":
                mock_data[k] = title
            elif k == "requester":
                mock_data[k] = requester
            elif k in ("total_amount", "total_approved_amount", "total_budget", "subtotal"):
                mock_data[k] = total_amount
            elif k in ("ref_doc_no", "ref_memo_no"):
                mock_data[k] = ref_doc_no
            elif k == "transfer_destination":
                mock_data[k] = transfer_dest
            elif k in ("expense_items", "approval_items", "line_items", "item_details"):
                mock_data[k] = expense_items
            elif k == "payer":
                mock_data[k] = requester
            elif k == "inspectors":
                mock_data[k] = requester
            elif k == "disbursement_type":
                mock_data[k] = "ทดรองจ่าย" if "ทดรองจ่าย" in ocr_markdown else ("เงินสดย่อย" if "เงินสดย่อย" in ocr_markdown else "ไม่มีระบุในเอกสาร")
            elif k == "vendor_name":
                mock_data[k] = title if title != "ไม่มีระบุในเอกสาร" else "ไม่มีระบุในเอกสาร"
            elif k == "vat":
                mock_data[k] = None
            else:
                mock_data[k] = "ไม่มีระบุในเอกสาร"

        type_name = DOCUMENT_TYPE_TITLES.get(document_type, document_type)
        mock_think = (
            f"1. สกัดข้อมูลตามประเภทเอกสารเป้าหมาย: {type_name}\n"
            f"2. สกัดฟิลด์สำคัญจากข้อความ OCR จริง: {list(mock_data.keys())}\n"
            f"3. ยอดรวมเงินที่ตรวจพบในเอกสาร: {total_amount} บาท"
        )

        return self._build_result_object(
            document_type=document_type,
            data=mock_data,
            thinking_process=mock_think,
            model_used=f"{model_name} (Dynamic OCR Fallback Mode)",
            latency_ms=elapsed_ms,
            is_mock=True,
            raw_llm_response=json.dumps(mock_data, ensure_ascii=False, indent=2)
        )
