"""
Comprehensive test suite for Component 4:
Data Validation, Normalization & Business Rules Engine (src/validator.py).
Executable directly with: python tests/test_validator.py
"""

import sys
from pathlib import Path
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.schemas import (
    DocumentType,
    ExpenseItem,
    TypeDirectedExtractionResult,
    ValidationSeverity,
)
from src.validator import FinancialDocumentValidator


def _compute_valid_tax_id(first_12_digits: str) -> str:
    """Helper to compute valid 13th check digit for testing."""
    digits = [int(ch) for ch in first_12_digits]
    s = sum(digits[i] * (13 - i) for i in range(12))
    check_digit = (11 - (s % 11)) % 10
    return first_12_digits + str(check_digit)


class TestFinancialDocumentValidator(unittest.TestCase):

    def setUp(self):
        self.validator = FinancialDocumentValidator(petty_cash_threshold=10000.0)

    # =========================================================================
    # 1. Thai Date Normalization Tests
    # =========================================================================

    def test_thai_date_full_buddhist_era(self):
        rep = self.validator.normalize_thai_date("18 กันยายน 2567")
        self.assertTrue(rep.is_valid)
        self.assertEqual(rep.iso_date, "2024-09-18")
        self.assertEqual(rep.thai_formatted, "18 กันยายน 2567")
        self.assertTrue(rep.is_buddhist_era)

    def test_thai_date_abbrev_buddhist_era(self):
        rep = self.validator.normalize_thai_date("18 ก.ย. 67")
        self.assertTrue(rep.is_valid)
        self.assertEqual(rep.iso_date, "2024-09-18")
        self.assertTrue(rep.is_buddhist_era)

    def test_thai_date_slash_be(self):
        rep = self.validator.normalize_thai_date("18/09/2567")
        self.assertTrue(rep.is_valid)
        self.assertEqual(rep.iso_date, "2024-09-18")
        self.assertTrue(rep.is_buddhist_era)

    def test_thai_date_christian_era(self):
        rep = self.validator.normalize_thai_date("30/06/2016")
        self.assertTrue(rep.is_valid)
        self.assertEqual(rep.iso_date, "2016-06-30")
        self.assertEqual(rep.thai_formatted, "30 มิถุนายน 2559")
        self.assertFalse(rep.is_buddhist_era)

    def test_thai_date_iso_format(self):
        rep = self.validator.normalize_thai_date("2024-09-18")
        self.assertTrue(rep.is_valid)
        self.assertEqual(rep.iso_date, "2024-09-18")
        self.assertEqual(rep.thai_formatted, "18 กันยายน 2567")

    def test_thai_digits_conversion(self):
        rep = self.validator.normalize_thai_date("๑๘ กันยายน ๒๕๖๗")
        self.assertTrue(rep.is_valid)
        self.assertEqual(rep.iso_date, "2024-09-18")

    def test_thai_date_prefix_cleaning(self):
        rep = self.validator.normalize_thai_date("วันที่ 5 มีนาคม 2566")
        self.assertTrue(rep.is_valid)
        self.assertEqual(rep.iso_date, "2023-03-05")

    def test_thai_date_invalid(self):
        rep = self.validator.normalize_thai_date("ข้อความที่ไม่ใช่วันที่")
        self.assertFalse(rep.is_valid)
        self.assertIsNone(rep.iso_date)

        rep_empty = self.validator.normalize_thai_date("")
        self.assertFalse(rep_empty.is_valid)

    # =========================================================================
    # 2. Thai 13-Digit Tax ID Validation Tests (Mod 11 Algorithm)
    # =========================================================================

    def test_valid_tax_id(self):
        valid_id = _compute_valid_tax_id("010555800123")
        rep = self.validator.validate_thai_tax_id(valid_id)
        self.assertTrue(rep.is_valid)
        self.assertEqual(rep.cleaned_id, valid_id)
        self.assertEqual(rep.formatted_id, f"{valid_id[0]}-{valid_id[1:5]}-{valid_id[5:10]}-{valid_id[10:12]}-{valid_id[12]}")
        self.assertIsNone(rep.error_message)

    def test_tax_id_with_hyphens_and_spaces(self):
        valid_id = _compute_valid_tax_id("010555800123")
        formatted_input = f" {valid_id[0]}-{valid_id[1:5]}-{valid_id[5:10]}-{valid_id[10:12]}-{valid_id[12]} "
        rep = self.validator.validate_thai_tax_id(formatted_input)
        self.assertTrue(rep.is_valid)
        self.assertEqual(rep.cleaned_id, valid_id)

    def test_tax_id_invalid_checksum(self):
        valid_id = _compute_valid_tax_id("010555800123")
        wrong_last_digit = "0" if valid_id[-1] != "0" else "1"
        invalid_id = valid_id[:-1] + wrong_last_digit

        rep = self.validator.validate_thai_tax_id(invalid_id)
        self.assertFalse(rep.is_valid)
        self.assertIn("Check digit", rep.error_message)

    def test_tax_id_invalid_length(self):
        rep = self.validator.validate_thai_tax_id("123456789")
        self.assertFalse(rep.is_valid)
        self.assertIn("ไม่ครบ 13 หลัก", rep.error_message)

    def test_tax_id_identical_repeats(self):
        rep = self.validator.validate_thai_tax_id("0000000000000")
        self.assertFalse(rep.is_valid)
        self.assertIn("ซ้ำกันทั้งหมด", rep.error_message)

    # =========================================================================
    # 3. Financial Arithmetic Reconciliation Tests
    # =========================================================================

    def test_math_reconcile_balanced_with_vat(self):
        items = [
            ExpenseItem(description="Item 1", total_price=50.0),
            ExpenseItem(description="Item 2", total_price=50.0),
        ]
        rep = self.validator.reconcile_financial_math(
            total_amount=107.0,
            subtotal=100.0,
            vat=7.0,
            items=items,
        )
        self.assertTrue(rep.is_balanced)
        self.assertEqual(rep.difference, 0.0)
        self.assertEqual(rep.expected_total, 107.0)
        self.assertEqual(rep.actual_total, 107.0)
        self.assertEqual(rep.line_items_sum, 100.0)

    def test_math_reconcile_unbalanced_mismatch(self):
        items = [ExpenseItem(description="Item 1", total_price=100.0)]
        rep = self.validator.reconcile_financial_math(
            total_amount=200.0,
            subtotal=100.0,
            vat=7.0,
            items=items,
        )
        self.assertFalse(rep.is_balanced)
        self.assertEqual(rep.difference, 93.0)

    def test_math_reconcile_line_items_only(self):
        items = [
            ExpenseItem(description="Item 1", quantity=2, unit_price=150.0),
            ExpenseItem(description="Item 2", total_price=200.0),
        ]
        rep = self.validator.reconcile_financial_math(
            total_amount=500.0,
            subtotal=None,
            vat=None,
            items=items,
        )
        self.assertTrue(rep.is_balanced)
        self.assertEqual(rep.line_items_sum, 500.0)

    # =========================================================================
    # 4. University Document Types & Business Rules Tests
    # =========================================================================

    def test_principle_approval_request_valid(self):
        res = TypeDirectedExtractionResult(
            document_type=DocumentType.PRINCIPLE_APPROVAL_REQUEST.value,
            document_type_name_th="1. เอกสารขออนุมัติหลักการ",
            fields={
                "doc_no": "มอ 0521.1.04/123",
                "doc_date": "18 กันยายน 2567",
                "title": "ขออนุมัติหลักการจัดโครงการสัมมนาวิชาการ",
                "requester": "ภาควิชาวิศวกรรมคอมพิวเตอร์",
                "total_amount": 25000.0,
            },
            expense_items=[
                ExpenseItem(description="ค่าอาหารว่างและเครื่องดื่ม", total_price=10000.0),
                ExpenseItem(description="ค่าตอบแทนวิทยากร", total_price=15000.0),
            ],
            total_amount=25000.0,
        )

        val_res = self.validator.validate(res)
        self.assertTrue(val_res.is_valid)
        self.assertEqual(val_res.status, "PASSED")
        self.assertEqual(val_res.date_report.iso_date, "2024-09-18")
        self.assertTrue(val_res.math_report.is_balanced)

    def test_principle_approval_request_missing_required(self):
        res = TypeDirectedExtractionResult(
            document_type=DocumentType.PRINCIPLE_APPROVAL_REQUEST.value,
            document_type_name_th="1. เอกสารขออนุมัติหลักการ",
            fields={
                "doc_no": "มอ 0521.1.04/123",
                "doc_date": "18 กันยายน 2567",
            },
            total_amount=None,
        )

        val_res = self.validator.validate(res)
        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.status, "ERROR")
        missing_fields = [issue.field for issue in val_res.issues if issue.code == "MISSING_REQUIRED_FIELD"]
        self.assertIn("title", missing_fields)
        self.assertIn("requester", missing_fields)
        self.assertIn("total_amount", missing_fields)

    def test_disbursement_petty_cash_threshold_warning(self):
        res = TypeDirectedExtractionResult(
            document_type=DocumentType.DISBURSEMENT_APPROVAL_REQUEST.value,
            document_type_name_th="3. ขออนุมัติเบิกจ่าย",
            fields={
                "doc_no": "มอ 0521/456",
                "doc_date": "2024-09-18",
                "title": "ขออนุมัติเบิกจ่ายเงินสดย่อยเพื่อซื้ออุปกรณ์เร่งด่วน",
                "disbursement_type": "เงินสดย่อย",
                "total_amount": 15000.0,
            },
            total_amount=15000.0,
        )

        val_res = self.validator.validate(res)
        self.assertTrue(val_res.is_valid)
        self.assertEqual(val_res.status, "WARNING")
        warning_codes = [i.code for i in val_res.issues if i.severity == ValidationSeverity.WARNING]
        self.assertIn("PETTY_CASH_EXCEEDS_LIMIT", warning_codes)

    def test_advance_payment_request_1_validation(self):
        res = TypeDirectedExtractionResult(
            document_type=DocumentType.ADVANCE_PAYMENT_REQUEST_1.value,
            document_type_name_th="4. แบบเบิกเงินทดรองจ่าย (แบบที่ 1)",
            fields={
                "doc_no": "ทจ 01/2567",
                "requester": "นายสมชาย ใจดี",
                "total_amount": 5000.0,
            },
            total_amount=5000.0,
        )

        val_res = self.validator.validate(res)
        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.status, "ERROR")
        missing_fields = [i.field for i in val_res.issues]
        self.assertIn("ref_doc_no", missing_fields)
        self.assertIn("transfer_destination", missing_fields)

    def test_parcel_inspection_validation(self):
        res = TypeDirectedExtractionResult(
            document_type=DocumentType.PARCEL_INSPECTION.value,
            document_type_name_th="7. ใบตรวจรับพัสดุ",
            fields={
                "doc_no": "พด 12/2567",
                "inspection_date": "18/09/2567",
                "inspectors": "1. นาย ก ประธานกรรมการ 2. นาย ข กรรมการ",
            },
        )

        val_res = self.validator.validate(res)
        self.assertTrue(val_res.is_valid)
        self.assertEqual(val_res.status, "PASSED")

    def test_general_receipt_full_validation(self):
        valid_tax_id = _compute_valid_tax_id("010555800123")
        res = TypeDirectedExtractionResult(
            document_type=DocumentType.GENERAL_RECEIPT.value,
            document_type_name_th="10. ใบเสร็จรับเงิน/ใบกำกับภาษีทั่วไป",
            fields={
                "vendor_name": "ร้านค้าพาเพลิน",
                "vendor_tax_id": valid_tax_id,
                "doc_date": "18 กันยายน 2567",
                "subtotal": 1000.0,
                "vat": 70.0,
                "total_amount": 1070.0,
            },
            expense_items=[
                ExpenseItem(description="กระดาษ A4", total_price=1000.0)
            ],
            subtotal=1000.0,
            vat=70.0,
            total_amount=1070.0,
        )

        val_res = self.validator.validate(res)
        self.assertTrue(val_res.is_valid)
        self.assertEqual(val_res.status, "PASSED")
        self.assertTrue(val_res.tax_id_report.is_valid)
        self.assertEqual(val_res.date_report.iso_date, "2024-09-18")
        self.assertTrue(val_res.math_report.is_balanced)
        self.assertEqual(val_res.normalized_data["tax_id_formatted"], f"{valid_tax_id[0]}-{valid_tax_id[1:5]}-{valid_tax_id[5:10]}-{valid_tax_id[10:12]}-{valid_tax_id[12]}")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Running Component 4 Validation & Rules Engine Test Suite...")
    print("=" * 60)
    unittest.main(verbosity=2)
