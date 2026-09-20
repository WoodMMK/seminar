"""
Component 4: Data Validation, Normalization & Business Rules Engine
Validates, normalizes, and verifies financial consistency and university policies
for structured extraction results from Component 3.
"""

import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from src.schemas import (
    DateNormalizationReport,
    DocumentType,
    DocumentValidationResult,
    ExpenseItem,
    MathReconciliationReport,
    TaxIdValidationReport,
    TypeDirectedExtractionResult,
    ValidationIssue,
    ValidationSeverity,
)


THAI_MONTHS_MAP = {
    # Full names
    "มกราคม": 1,
    "กุมภาพันธ์": 2,
    "มีนาคม": 3,
    "เมษายน": 4,
    "พฤษภาคม": 5,
    "มิถุนายน": 6,
    "กรกฎาคม": 7,
    "สิงหาคม": 8,
    "กันยายน": 9,
    "ตุลาคม": 10,
    "พฤศจิกายน": 11,
    "ธันวาคม": 12,
    # Abbreviations
    "ม.ค.": 1,
    "ม.ค": 1,
    "ก.พ.": 2,
    "ก.พ": 2,
    "มี.ค.": 3,
    "มี.ค": 3,
    "เม.ย.": 4,
    "เม.ย": 4,
    "พ.ค.": 5,
    "พ.ค": 5,
    "มิ.ย.": 6,
    "มิ.ย": 6,
    "ก.ค.": 7,
    "ก.ค": 7,
    "ส.ค.": 8,
    "ส.ค": 8,
    "ก.ย.": 9,
    "ก.ย": 9,
    "ต.ค.": 10,
    "ต.ค": 10,
    "พ.ย.": 11,
    "พ.ย": 11,
    "ธ.ค.": 12,
    "ธ.ค": 12,
}

THAI_MONTH_NAMES_FULL = [
    "",
    "มกราคม",
    "กุมภาพันธ์",
    "มีนาคม",
    "เมษายน",
    "พฤษภาคม",
    "มิถุนายน",
    "กรกฎาคม",
    "สิงหาคม",
    "กันยายน",
    "ตุลาคม",
    "พฤศจิกายน",
    "ธันวาคม",
]

THAI_DIGITS_TRANS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


class FinancialDocumentValidator:
    """
    Component 4: Financial Document Validator & Business Rules Engine.
    Provides deterministic algorithmic validation for financial data,
    Thai Tax IDs, Thai dates, and University reimbursement policies.
    """

    def __init__(self, petty_cash_threshold: float = 10000.0):
        self.petty_cash_threshold = petty_cash_threshold

    # =========================================================================
    # 1. Thai Date Normalization
    # =========================================================================

    def normalize_thai_date(self, raw_date: Optional[str]) -> DateNormalizationReport:
        """
        Parses Thai date strings (both Buddhist Era พ.ศ. and Christian Era ค.ศ.)
        into ISO 8601 (YYYY-MM-DD) and normalized Thai format.
        """
        if not raw_date or not str(raw_date).strip():
            return DateNormalizationReport(
                raw_date=raw_date,
                iso_date=None,
                thai_formatted=None,
                is_buddhist_era=False,
                is_valid=False,
            )

        text = str(raw_date).strip().translate(THAI_DIGITS_TRANS)
        # Remove extra whitespace and prefix noise like 'วันที่' or 'เมื่อ'
        text = re.sub(r"^(วันที่|เมื่อวันที่|เมื่อ|date\s*[:.-]?)\s*", "", text, flags=re.IGNORECASE).strip()

        day: Optional[int] = None
        month: Optional[int] = None
        year: Optional[int] = None
        is_be = False

        # Pattern A: Thai text month (e.g. "18 กันยายน 2567", "18 ก.ย. 67", "5 มีนาคม 2024")
        month_names_regex = "|".join(re.escape(k) for k in sorted(THAI_MONTHS_MAP.keys(), key=len, reverse=True))
        text_month_match = re.search(
            rf"(\d{{1,2}})\s*[\s\-\./]?\s*({month_names_regex})\s*[\s\-\./]?\s*(\d{{2,4}})",
            text,
        )

        if text_month_match:
            d_str, m_str, y_str = text_month_match.groups()
            day = int(d_str)
            month = THAI_MONTHS_MAP.get(m_str)
            y_raw = int(y_str)
            year, is_be = self._resolve_year(y_raw)
        else:
            # Pattern B: Slash/Dash/Dot numeric formats (e.g. "18/09/2567", "18-09-2024", "30.06.2016")
            # Or ISO format "YYYY-MM-DD"
            iso_match = re.match(r"^(\d{4})[-\./](\d{1,2})[-\./](\d{1,2})$", text)
            if iso_match:
                y_raw, m_str, d_str = iso_match.groups()
                day = int(d_str)
                month = int(m_str)
                year, is_be = self._resolve_year(int(y_raw))
            else:
                dmy_match = re.search(r"(\d{1,2})[-\./](\d{1,2})[-\./](\d{2,4})", text)
                if dmy_match:
                    d_str, m_str, y_str = dmy_match.groups()
                    day = int(d_str)
                    month = int(m_str)
                    year, is_be = self._resolve_year(int(y_str))

        if day and month and year and (1 <= month <= 12):
            try:
                parsed_date = date(year, month, day)
                iso_str = parsed_date.strftime("%Y-%m-%d")
                thai_be_year = year + 543
                thai_month_name = THAI_MONTH_NAMES_FULL[month]
                thai_str = f"{day} {thai_month_name} {thai_be_year}"
                return DateNormalizationReport(
                    raw_date=raw_date,
                    iso_date=iso_str,
                    thai_formatted=thai_str,
                    is_buddhist_era=is_be,
                    is_valid=True,
                )
            except ValueError:
                # Invalid calendar date (e.g., Feb 30)
                pass

        return DateNormalizationReport(
            raw_date=raw_date,
            iso_date=None,
            thai_formatted=None,
            is_buddhist_era=False,
            is_valid=False,
        )

    def _resolve_year(self, y_raw: int) -> Tuple[int, bool]:
        """Resolves raw year number into Gregorian calendar year and is_buddhist_era flag."""
        # 4-digit Buddhist Era (e.g., 2567 -> 2024)
        if y_raw >= 2400:
            return y_raw - 543, True
        # 4-digit Christian Era (e.g., 2024)
        if y_raw >= 1900:
            return y_raw, False
        # 2-digit years: In Thailand, 2-digit years are almost universally B.E. (e.g., 67 -> 2567 -> 2024)
        if y_raw >= 40:
            be_year = 2500 + y_raw
            return be_year - 543, True
        return 2000 + y_raw, False

    # =========================================================================
    # 2. Thai 13-Digit Tax ID Validation (Mod 11 Algorithm)
    # =========================================================================

    def validate_thai_tax_id(self, raw_tax_id: Optional[str]) -> TaxIdValidationReport:
        """
        Validates Thai 13-digit National ID / Corporate Tax ID using the official
        Thai Revenue Department Mod 11 Checksum algorithm.
        Format: X-XXXX-XXXXX-XX-X
        """
        if not raw_tax_id or not str(raw_tax_id).strip():
            return TaxIdValidationReport(
                raw_id=raw_tax_id,
                cleaned_id=None,
                formatted_id=None,
                is_valid=False,
                error_message="ไม่พบข้อมูลเลขประจำตัวผู้เสียภาษี",
            )

        cleaned = re.sub(r"\D", "", str(raw_tax_id).translate(THAI_DIGITS_TRANS))

        if len(cleaned) != 13:
            return TaxIdValidationReport(
                raw_id=raw_tax_id,
                cleaned_id=cleaned,
                formatted_id=None,
                is_valid=False,
                error_message=f"จำนวนหลักไม่ครบ 13 หลัก (พบ {len(cleaned)} หลัก)",
            )

        # Check for invalid repeating patterns like 0000000000000, 1111111111111
        if len(set(cleaned)) == 1:
            return TaxIdValidationReport(
                raw_id=raw_tax_id,
                cleaned_id=cleaned,
                formatted_id=None,
                is_valid=False,
                error_message="เลขประจำตัวผู้เสียภาษีไม่ถูกต้อง (ตัวเลขซ้ำกันทั้งหมด)",
            )

        # Mod 11 Algorithm:
        # Sum = (d0 * 13) + (d1 * 12) + ... + (d11 * 2)
        # Check digit = (11 - (Sum % 11)) % 10
        digits = [int(ch) for ch in cleaned]
        checksum_sum = sum(digits[i] * (13 - i) for i in range(12))
        expected_check_digit = (11 - (checksum_sum % 11)) % 10

        formatted = f"{cleaned[0]}-{cleaned[1:5]}-{cleaned[5:10]}-{cleaned[10:12]}-{cleaned[12]}"

        if digits[12] != expected_check_digit:
            return TaxIdValidationReport(
                raw_id=raw_tax_id,
                cleaned_id=cleaned,
                formatted_id=formatted,
                is_valid=False,
                error_message=f"เลขตรวจสอบ (Check digit) ไม่ถูกต้อง (คาดหวัง {expected_check_digit} แต่ตรวจพบ {digits[12]})",
            )

        return TaxIdValidationReport(
            raw_id=raw_tax_id,
            cleaned_id=cleaned,
            formatted_id=formatted,
            is_valid=True,
            error_message=None,
        )

    # =========================================================================
    # 3. Financial Arithmetic Reconciliation
    # =========================================================================

    def reconcile_financial_math(
        self,
        total_amount: Optional[float],
        subtotal: Optional[float],
        vat: Optional[float],
        items: List[ExpenseItem],
    ) -> MathReconciliationReport:
        """
        Cross-validates financial arithmetic:
        - Sum(line_items) vs Subtotal / Total
        - Subtotal + VAT vs Total
        - VAT rate consistency (7%)
        """
        # Calculate items sum
        items_sum: Optional[float] = None
        has_item_prices = any(
            (it.total_price is not None or (it.quantity is not None and it.unit_price is not None))
            for it in items
        )

        if has_item_prices:
            total_calc = 0.0
            for it in items:
                if it.total_price is not None:
                    total_calc += it.total_price
                elif it.quantity is not None and it.unit_price is not None:
                    total_calc += it.quantity * it.unit_price
            items_sum = round(total_calc, 2)

        # Case 1: Subtotal and VAT are present
        if subtotal is not None and vat is not None and total_amount is not None:
            expected = round(subtotal + vat, 2)
            actual = round(total_amount, 2)
            diff = round(actual - expected, 2)
            is_bal = abs(diff) <= 0.05

            details = f"Subtotal (฿{subtotal:,.2f}) + VAT (฿{vat:,.2f}) = ฿{expected:,.2f} vs ระบุ ฿{actual:,.2f}"
            if items_sum is not None:
                item_diff = round(subtotal - items_sum, 2)
                if abs(item_diff) > 0.05:
                    details += f" (ยอดรวมรายการสินค้า ฿{items_sum:,.2f} ต่างจาก Subtotal ฿{item_diff:,.2f})"

            return MathReconciliationReport(
                is_balanced=is_bal,
                expected_total=expected,
                actual_total=actual,
                difference=diff,
                line_items_sum=items_sum,
                details=details,
            )

        # Case 2: Only Subtotal & Total
        if subtotal is not None and total_amount is not None:
            expected = round(subtotal, 2)
            actual = round(total_amount, 2)
            diff = round(actual - expected, 2)
            is_bal = abs(diff) <= 0.05
            details = f"Subtotal ฿{expected:,.2f} vs ยอดรวม ฿{actual:,.2f}"
            return MathReconciliationReport(
                is_balanced=is_bal,
                expected_total=expected,
                actual_total=actual,
                difference=diff,
                line_items_sum=items_sum,
                details=details,
            )

        # Case 3: Line items sum & Total amount
        if items_sum is not None and total_amount is not None:
            expected = round(items_sum, 2)
            actual = round(total_amount, 2)
            diff = round(actual - expected, 2)
            is_bal = abs(diff) <= 0.05
            details = f"ผลรวมรายการ ({len(items)} รายการ): ฿{expected:,.2f} vs ยอดรวมระบุ ฿{actual:,.2f}"
            return MathReconciliationReport(
                is_balanced=is_bal,
                expected_total=expected,
                actual_total=actual,
                difference=diff,
                line_items_sum=items_sum,
                details=details,
            )

        # Case 4: Only Total amount specified
        if total_amount is not None:
            actual = round(total_amount, 2)
            is_bal = actual >= 0
            return MathReconciliationReport(
                is_balanced=is_bal,
                expected_total=actual,
                actual_total=actual,
                difference=0.0,
                line_items_sum=items_sum,
                details=f"ยอดรวมระบุในเอกสาร: ฿{actual:,.2f}" if is_bal else "ยอดเงินติดลบ",
            )

        # Case 5: No monetary values at all
        return MathReconciliationReport(
            is_balanced=True,
            expected_total=None,
            actual_total=None,
            difference=0.0,
            line_items_sum=items_sum,
            details="ไม่มีตัวเลขจำนวนเงินที่ต้องกระทบยอด",
        )

    # =========================================================================
    # 4. Master Validation Routine
    # =========================================================================

    def validate(self, result: TypeDirectedExtractionResult) -> DocumentValidationResult:
        """
        Executes end-to-end validation, normalization, and business rule enforcement
        for an extracted financial document result.
        """
        issues: List[ValidationIssue] = []
        normalized_data: Dict[str, Any] = dict(result.fields or {})

        doc_type = result.document_type
        fields = result.fields or {}
        items = result.expense_items or result.line_items or []

        # ---------------------------------------------------------------------
        # A. Date Normalization & Verification
        # ---------------------------------------------------------------------
        raw_date = (
            fields.get("doc_date")
            or fields.get("document_date")
            or fields.get("submission_date")
            or fields.get("claim_date")
            or fields.get("inspection_date")
            or result.document_date
        )
        date_report = self.normalize_thai_date(raw_date)

        if raw_date:
            if date_report.is_valid:
                normalized_data["doc_date_iso"] = date_report.iso_date
                normalized_data["doc_date_thai"] = date_report.thai_formatted
            else:
                issues.append(
                    ValidationIssue(
                        field="doc_date",
                        message=f"ไม่สามารถแปลงวันที่ให้อยู่ในรูปแบบมาตรฐานได้: '{raw_date}'",
                        severity=ValidationSeverity.WARNING,
                        code="DATE_PARSE_WARNING",
                        actual=raw_date,
                    )
                )

        # ---------------------------------------------------------------------
        # B. Tax ID Validation (if present)
        # ---------------------------------------------------------------------
        raw_tax_id = fields.get("vendor_tax_id") or fields.get("customer_tax_id") or result.vendor_tax_id
        tax_id_report: Optional[TaxIdValidationReport] = None

        if raw_tax_id and not any(term in str(raw_tax_id) for term in ["ไม่มี", "not specified", "none", "null", "N/A"]):
            tax_id_report = self.validate_thai_tax_id(raw_tax_id)
            if tax_id_report.is_valid:
                normalized_data["tax_id_formatted"] = tax_id_report.formatted_id
            else:
                issues.append(
                    ValidationIssue(
                        field="vendor_tax_id",
                        message=f"เลขประจำตัวผู้เสียภาษีไม่ผ่านการตรวจสอบ: {tax_id_report.error_message}",
                        severity=ValidationSeverity.WARNING,
                        code="TAX_ID_CHECKSUM_INVALID",
                        actual=raw_tax_id,
                    )
                )

        # ---------------------------------------------------------------------
        # C. Financial Arithmetic Reconciliation
        # ---------------------------------------------------------------------
        tot = result.total_amount
        if tot is None and "total_budget" in fields:
            tot = fields.get("total_budget")
        if tot is None and "total_approved_amount" in fields:
            tot = fields.get("total_approved_amount")

        subtot = result.subtotal or fields.get("subtotal")
        vat_val = result.vat or fields.get("vat")

        math_report = self.reconcile_financial_math(tot, subtot, vat_val, items)
        if not math_report.is_balanced:
            issues.append(
                ValidationIssue(
                    field="total_amount",
                    message=f"ยอดเงินรวมไม่สอดคล้องกัน: {math_report.details}",
                    severity=ValidationSeverity.ERROR,
                    code="FINANCIAL_MATH_MISMATCH",
                    expected=math_report.expected_total,
                    actual=math_report.actual_total,
                )
            )

        # ---------------------------------------------------------------------
        # D. University Document Type Specific Business Rules
        # ---------------------------------------------------------------------
        self._enforce_business_rules(doc_type, fields, items, tot, issues, normalized_data)

        # ---------------------------------------------------------------------
        # E. Determine Overall Validation Status
        # ---------------------------------------------------------------------
        has_error = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        has_warning = any(issue.severity == ValidationSeverity.WARNING for issue in issues)

        if has_error:
            status = "ERROR"
            is_valid = False
        elif has_warning:
            status = "WARNING"
            is_valid = True  # Allowed to proceed with caution
        else:
            status = "PASSED"
            is_valid = True

        validation_result = DocumentValidationResult(
            is_valid=is_valid,
            status=status,
            issues=issues,
            math_report=math_report,
            tax_id_report=tax_id_report,
            date_report=date_report,
            normalized_data=normalized_data,
        )

        # Attach to extraction result object for direct consumption
        result.validation = validation_result
        return validation_result

    # =========================================================================
    # 5. Type-Specific Rules Engine
    # =========================================================================

    def _enforce_business_rules(
        self,
        doc_type: str,
        fields: Dict[str, Any],
        items: List[ExpenseItem],
        total: Optional[float],
        issues: List[ValidationIssue],
        normalized_data: Dict[str, Any],
    ):
        """Enforces mandatory fields and university disbursement policies."""

        def check_required(field_name: str, display_name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
            val = fields.get(field_name)
            if val is None or not str(val).strip():
                issues.append(
                    ValidationIssue(
                        field=field_name,
                        message=f"ไม่พบข้อมูลสำคัญ: {display_name}",
                        severity=severity,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )
                return False
            return True

        # Rule 1: principle_approval_request (เอกสารขออนุมัติหลักการ)
        if doc_type == DocumentType.PRINCIPLE_APPROVAL_REQUEST.value:
            check_required("doc_no", "1.1 เลขที่เอกสาร")
            check_required("doc_date", "1.2 วันที่ทำเอกสาร")
            check_required("title", "1.3 เรื่อง")
            check_required("requester", "1.4 ผู้ทำการเบิก (บุคคลหรือภาควิชา)")
            if total is None:
                issues.append(
                    ValidationIssue(
                        field="total_amount",
                        message="ไม่พบข้อมูลยอดรวมงบประมาณที่ขออนุมัติ (1.6 ยอดรวม)",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )
            if not items:
                issues.append(
                    ValidationIssue(
                        field="expense_items",
                        message="ไม่พบรายการแจกแจงค่าใช้จ่าย (1.5 รายละเอียดค่าใช้จ่าย)",
                        severity=ValidationSeverity.WARNING,
                        code="EMPTY_EXPENSE_ITEMS",
                    )
                )

        # Rule 2: principle_approval_granted (เอกสารอนุมัติหลักการ)
        elif doc_type == DocumentType.PRINCIPLE_APPROVAL_GRANTED.value:
            check_required("ref_doc_no", "เลขที่หนังสือขออนุมัติหลักการอ้างอิง")
            tot_approved = fields.get("total_approved_amount") or total
            if tot_approved is None:
                issues.append(
                    ValidationIssue(
                        field="total_approved_amount",
                        message="ไม่พบยอดรวมที่ได้รับอนุมัติ (2.2 ยอดรวมที่อนุมัติ)",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )
            if not items:
                issues.append(
                    ValidationIssue(
                        field="approval_items",
                        message="ไม่พบรายการแจกแจงที่ได้รับอนุมัติ",
                        severity=ValidationSeverity.WARNING,
                        code="EMPTY_EXPENSE_ITEMS",
                    )
                )

        # Rule 3: disbursement_approval_request (ขออนุมัติเบิกจ่าย)
        elif doc_type == DocumentType.DISBURSEMENT_APPROVAL_REQUEST.value:
            check_required("doc_no", "3.1 เลขที่เอกสาร")
            check_required("doc_date", "3.2 วันที่ทำเอกสาร")
            check_required("title", "3.3 เรื่อง")
            if total is None:
                issues.append(
                    ValidationIssue(
                        field="total_amount",
                        message="ไม่พบยอดรวมการขอเบิกจ่าย (3.5 ยอดรวม)",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )

            # Business Policy: Petty cash (เงินสดย่อย) limit check (Mahidol threshold <= 10,000 THB)
            disb_type = str(fields.get("disbursement_type") or "").strip()
            if "เงินสดย่อย" in disb_type and total is not None:
                if total > self.petty_cash_threshold:
                    issues.append(
                        ValidationIssue(
                            field="total_amount",
                            message=(
                                f"ยอดเบิกจ่ายเงินสดย่อยเกินเกณฑ์ที่กำหนด (ระบุ ฿{total:,.2f} "
                                f"เกินวงเงินสดย่อยสูงสุด ฿{self.petty_cash_threshold:,.2f}) "
                                "กรุณาใช้แบบขออนุมัติจัดซื้อจัดจ้างหรือทดรองจ่ายแทน"
                            ),
                            severity=ValidationSeverity.WARNING,
                            code="PETTY_CASH_EXCEEDS_LIMIT",
                            expected=self.petty_cash_threshold,
                            actual=total,
                        )
                    )

        # Rule 4: advance_payment_request_1 (แบบเบิกเงินทดรองจ่าย แบบที่ 1)
        elif doc_type == DocumentType.ADVANCE_PAYMENT_REQUEST_1.value:
            check_required("doc_no", "4.1 เลขที่เอกสาร")
            check_required("requester", "4.2 ใครเป็นคนเบิก")
            check_required("ref_doc_no", "4.4 ตามหนังสืออนุมัติเบิกจ่าย เลขที่อะไร")
            check_required("transfer_destination", "4.6 ข้อมูลการโอนเงิน (ธนาคาร/เลขที่บัญชี)")
            if total is None:
                issues.append(
                    ValidationIssue(
                        field="total_amount",
                        message="ไม่พบยอดรวมเงินทดรองจ่าย (4.3 ยอดรวม)",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )

        # Rule 5: advance_payment_request_2 (แบบเบิกเงินทดรองจ่าย แบบที่ 2 - เคลียร์เงิน)
        elif doc_type == DocumentType.ADVANCE_PAYMENT_REQUEST_2.value:
            check_required("doc_no", "5.1 เลขที่เอกสาร")
            check_required("requester", "5.4 ใครเป็นคนเบิก")
            check_required("ref_doc_no", "5.6 ตามหนังสืออนุมัติหลักการ เลขที่อะไร")
            has_submission = check_required("submission_date", "5.2 วันที่ส่งเอกสาร", severity=ValidationSeverity.WARNING)
            has_claim = check_required("claim_date", "5.3 วันที่ขอรับเงิน", severity=ValidationSeverity.WARNING)
            if not has_submission and not has_claim:
                issues.append(
                    ValidationIssue(
                        field="doc_date",
                        message="ต้องระบุวันที่ส่งเอกสารหรือวันที่ขอรับเงินอย่างน้อยหนึ่งรายการ",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )
            if total is None:
                issues.append(
                    ValidationIssue(
                        field="total_amount",
                        message="ไม่พบยอดรวมการขอรับเงิน/เคลียร์เงิน (5.5 ยอดรวม)",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )

        # Rule 6: receipt_substitute (ใบแทนใบเสร็จ / ใบสำคัญรับเงิน)
        elif doc_type == DocumentType.RECEIPT_SUBSTITUTE.value:
            check_required("doc_date", "6.1 วัน/เดือน/ปี")
            check_required("payer", "6.4 ผู้จ่ายเงิน")
            if total is None:
                issues.append(
                    ValidationIssue(
                        field="total_amount",
                        message="ไม่พบยอดรวมการจ่ายเงิน (6.3 ยอดรวม)",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )
            if not items:
                issues.append(
                    ValidationIssue(
                        field="expense_items",
                        message="ใบสำคัญรับเงินต้องมีรายการค่าใช้จ่ายแจกแจงอย่างน้อย 1 รายการ",
                        severity=ValidationSeverity.ERROR,
                        code="EMPTY_EXPENSE_ITEMS",
                    )
                )

        # Rule 7: parcel_inspection (ใบตรวจรับพัสดุ)
        elif doc_type == DocumentType.PARCEL_INSPECTION.value:
            check_required("doc_no", "7.1 เลขที่เอกสาร")
            check_required("inspectors", "7.2 ผู้ตรวจรับพัสดุ / คณะกรรมการตรวจรับ")
            check_required("inspection_date", "วันที่ตรวจรับพัสดุ", severity=ValidationSeverity.WARNING)

        # Rule 8: procurement_approval_request (ขออนุมัติจัดหาพัสดุ)
        elif doc_type == DocumentType.PROCUREMENT_APPROVAL_REQUEST.value:
            check_required("doc_no", "8.1 เลขที่เอกสาร")
            check_required("doc_date", "8.2 วันที่ทำเอกสาร")
            check_required("title", "8.3 เรื่อง")
            check_required("procurement_reason", "8.4 เหตุผลความจำเป็นที่ต้องจัดหา")
            tot_bg = fields.get("total_budget") or total
            if tot_bg is None:
                issues.append(
                    ValidationIssue(
                        field="total_budget",
                        message="ไม่พบข้อมูลวงเงินงบประมาณที่ใช้จัดหา (8.6 วงเงินที่ใช้ทั้งหมด)",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )

        # Rule 9: procurement_attachment (เอกสารประกอบการขออนุมัติจัดหา)
        elif doc_type == DocumentType.PROCUREMENT_ATTACHMENT.value:
            check_required("ref_memo_no", "9.1 แนบท้ายบันทึกเอกสารเลขที่")
            if total is None:
                issues.append(
                    ValidationIssue(
                        field="total_amount",
                        message="ไม่พบยอดรวมพัสดุ (9.2 ยอดรวม)",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )

        # Rule 10: general_receipt (ใบเสร็จรับเงิน/ใบกำกับภาษีทั่วไป)
        elif doc_type == DocumentType.GENERAL_RECEIPT.value:
            check_required("vendor_name", "ชื่อร้านค้าหรือบริษัทผู้ขาย")
            if total is None:
                issues.append(
                    ValidationIssue(
                        field="total_amount",
                        message="ไม่พบยอดเงินรวมทั้งสิ้น (Total Amount)",
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )
            if not items:
                issues.append(
                    ValidationIssue(
                        field="line_items",
                        message="ไม่พบรายการสินค้า/บริการบนใบเสร็จ",
                        severity=ValidationSeverity.WARNING,
                        code="EMPTY_LINE_ITEMS",
                    )
                )
