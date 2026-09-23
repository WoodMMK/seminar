"""
Data schemas for Financial Document Information Extraction (Component 3)
Defines structured contracts for 9 university reimbursement document types
as well as general commercial receipts.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PRINCIPLE_APPROVAL_REQUEST = "principle_approval_request"      # 1. เอกสารขออนุมัติหลักการ
    PRINCIPLE_APPROVAL_GRANTED = "principle_approval_granted"      # 2. เอกสารอนุมัติหลักการ
    DISBURSEMENT_APPROVAL_REQUEST = "disbursement_approval_request" # 3. ขออนุมัติเบิกจ่าย
    ADVANCE_PAYMENT_REQUEST_1 = "advance_payment_request_1"        # 4. แบบเบิกเงินทดรองจ่าย (แบบที่ 1)
    ADVANCE_PAYMENT_REQUEST_2 = "advance_payment_request_2"        # 5. แบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์)
    RECEIPT_SUBSTITUTE = "receipt_substitute"                      # 6. ใบแทนใบเสร็จ / ใบสำคัญรับเงิน
    PARCEL_INSPECTION = "parcel_inspection"                        # 7. ใบตรวจรับพัสดุ
    PROCUREMENT_APPROVAL_REQUEST = "procurement_approval_request"  # 8. ขออนุมัติจัดหาพัสดุ
    PROCUREMENT_ATTACHMENT = "procurement_attachment"              # 9. เอกสารประกอบการขออนุมัติจัดหา
    GENERAL_RECEIPT = "general_receipt"                            # 10. ใบเสร็จรับเงิน/ใบกำกับภาษีทั่วไป


DOCUMENT_TYPE_TITLES: Dict[str, str] = {
    DocumentType.PRINCIPLE_APPROVAL_REQUEST.value: "เอกสารขออนุมัติหลักการ",
    DocumentType.PRINCIPLE_APPROVAL_GRANTED.value: "เอกสารอนุมัติหลักการ",
    DocumentType.DISBURSEMENT_APPROVAL_REQUEST.value: "ขออนุมัติเบิกจ่าย",
    DocumentType.ADVANCE_PAYMENT_REQUEST_1.value: "แบบเบิกเงินทดรองจ่าย (สัญญายืมเงิน/เบิก)",
    DocumentType.ADVANCE_PAYMENT_REQUEST_2.value: "แบบเบิกเงินทดรองจ่าย (ขอรับเงิน/เคลียร์เงิน)",
    DocumentType.RECEIPT_SUBSTITUTE.value: "ใบแทนใบเสร็จ / ใบสำคัญรับเงิน",
    DocumentType.PARCEL_INSPECTION.value: "ใบตรวจรับพัสดุ",
    DocumentType.PROCUREMENT_APPROVAL_REQUEST.value: "ขออนุมัติจัดหาพัสดุ",
    DocumentType.PROCUREMENT_ATTACHMENT.value: "เอกสารประกอบการขออนุมัติจัดหา",
    DocumentType.GENERAL_RECEIPT.value: "ใบเสร็จรับเงิน / ใบกำกับภาษีทั่วไป (ร้านค้า/บริษัท)"
}


class ExpenseItem(BaseModel):
    """Details for an itemized expenditure or parcel item."""
    item_no: Optional[int] = Field(default=None, description="ลำดับรายการ เช่น 1, 2")
    description: str = Field(description="ชื่อรายการสินค้า บริการ หรือพัสดุ")
    quantity: Optional[float] = Field(default=None, description="จำนวน")
    unit: Optional[str] = Field(default=None, description="หน่วยนับ เช่น รีม, เล่ม, ครั้ง, ชิ้น")
    unit_price: Optional[float] = Field(default=None, description="ราคาต่อหน่วย (บาท)")
    total_price: Optional[float] = Field(default=None, description="จำนวนเงินรวม (บาท)")


# Alias for backward compatibility
LineItem = ExpenseItem


class PrincipleApprovalRequestData(BaseModel):
    """1. เอกสารขออนุมัติหลักการ"""
    doc_no: Optional[str] = Field(default=None, description="1.1 เลขที่เอกสาร")
    doc_date: Optional[str] = Field(default=None, description="1.2 วันที่ทำเอกสาร")
    title: Optional[str] = Field(default=None, description="1.3 เรื่อง")
    requester: Optional[str] = Field(default=None, description="1.4 ผู้ทำการเบิก (บุคคล หรือ ภาควิชา)")
    expense_items: List[ExpenseItem] = Field(default_factory=list, description="1.5 รายละเอียดค่าใช้จ่าย (เบิกอะไรบ้าง และเท่าไหร่บ้าง)")
    total_amount: Optional[float] = Field(default=None, description="1.6 ยอดรวม")


class PrincipleApprovalGrantedData(BaseModel):
    """2. เอกสารอนุมัติหลักการ"""
    ref_doc_no: Optional[str] = Field(default=None, description="ตามหนังสือขออนุมัติเลขที่")
    approval_items: List[ExpenseItem] = Field(default_factory=list, description="2.1 รายละเอียดการอนุมัติ (อนุมัติอะไรบ้าง เท่าไหร่บ้าง)")
    total_approved_amount: Optional[float] = Field(default=None, description="2.2 ยอดรวมที่อนุมัติ")


class DisbursementApprovalRequestData(BaseModel):
    """3. ขออนุมัติเบิกจ่าย"""
    doc_no: Optional[str] = Field(default=None, description="3.1 เลขที่เอกสาร")
    doc_date: Optional[str] = Field(default=None, description="3.2 วันที่ทำเอกสาร")
    title: Optional[str] = Field(default=None, description="3.3 เรื่อง")
    expense_items: List[ExpenseItem] = Field(default_factory=list, description="3.4 รายละเอียดค่าใช้จ่าย (เบิกอะไรบ้าง และเท่าไหร่บ้าง)")
    total_amount: Optional[float] = Field(default=None, description="3.5 ยอดรวม")
    disbursement_type: Optional[str] = Field(default=None, description="3.6 ประเภทของการเบิกจ่าย (เงินสดย่อย / ทดรองจ่าย)")


class AdvancePaymentRequest1Data(BaseModel):
    """4. แบบเบิกเงินทดรองจ่าย (แบบที่ 1)"""
    doc_no: Optional[str] = Field(default=None, description="4.1 เลขที่เอกสาร")
    requester: Optional[str] = Field(default=None, description="4.2 ใครเป็นคนเบิก")
    total_amount: Optional[float] = Field(default=None, description="4.3 ยอดรวม")
    ref_doc_no: Optional[str] = Field(default=None, description="4.4 ตามหนังสืออนุมัติเบิกจ่าย เลขที่อะไร")
    expense_items: List[ExpenseItem] = Field(default_factory=list, description="4.5 รายละเอียดตามประเภทค่าใช้จ่าย")
    transfer_destination: Optional[str] = Field(default=None, description="4.6 โอนเงินไปที่ใด (ธนาคาร/เลขบัญชี/ผู้รับ)")


class AdvancePaymentRequest2Data(BaseModel):
    """5. แบบเบิกเงินทดรองจ่าย (แบบที่ 2 - ขอรับเงิน/เคลียร์เงิน)"""
    doc_no: Optional[str] = Field(default=None, description="5.1 เลขที่เอกสาร")
    submission_date: Optional[str] = Field(default=None, description="5.2 วันที่ส่งเอกสาร")
    claim_date: Optional[str] = Field(default=None, description="5.3 วันที่ขอรับเงิน")
    requester: Optional[str] = Field(default=None, description="5.4 ใครเป็นคนเบิก")
    total_amount: Optional[float] = Field(default=None, description="5.5 ยอดรวม")
    ref_doc_no: Optional[str] = Field(default=None, description="5.6 ตามหนังสืออนุมัติหลักการ เลขที่อะไร")
    expense_items: List[ExpenseItem] = Field(default_factory=list, description="5.7 รายละเอียด")
    transfer_destination: Optional[str] = Field(default=None, description="5.8 โอนเงินไปที่ใด")


class ReceiptSubstituteData(BaseModel):
    """6. ใบแทนใบเสร็จ / ใบสำคัญรับเงิน"""
    doc_date: Optional[str] = Field(default=None, description="6.1 วัน/เดือน/ปี")
    expense_items: List[ExpenseItem] = Field(default_factory=list, description="6.2 รายละเอียดของรายการการเบิก")
    total_amount: Optional[float] = Field(default=None, description="6.3 ยอดรวม")
    payer: Optional[str] = Field(default=None, description="6.4 ผู้จ่ายเงิน")


class ParcelInspectionData(BaseModel):
    """7. ใบตรวจรับพัสดุ"""
    doc_no: Optional[str] = Field(default=None, description="7.1 เลขที่เอกสาร")
    inspectors: Optional[str] = Field(default=None, description="7.2 ผู้ตรวจรับพัสดุ (รายชื่อคณะกรรมการ/ผู้ตรวจรับ)")
    inspection_date: Optional[str] = Field(default=None, description="วันที่ตรวจรับ")
    ref_doc_no: Optional[str] = Field(default=None, description="ตามใบสั่งซื้อ/สัญญาเลขที่")


class ProcurementApprovalRequestData(BaseModel):
    """8. ขออนุมัติจัดหาพัสดุ"""
    doc_no: Optional[str] = Field(default=None, description="8.1 เลขที่เอกสาร")
    doc_date: Optional[str] = Field(default=None, description="8.2 วันที่ทำเอกสาร")
    title: Optional[str] = Field(default=None, description="8.3 เรื่อง")
    procurement_reason: Optional[str] = Field(default=None, description="8.4 เหตุผลที่ต้องจัดหา")
    item_details: List[ExpenseItem] = Field(default_factory=list, description="8.5 รายละเอียดของพัสดุ")
    total_budget: Optional[float] = Field(default=None, description="8.6 วงเงินที่ใช้ทั้งหมด")
    required_date: Optional[str] = Field(default=None, description="8.7 เวลาที่ต้องใช้พัสดุ")


class ProcurementAttachmentData(BaseModel):
    """9. เอกสารประกอบการขออนุมัติจัดหา"""
    ref_memo_no: Optional[str] = Field(default=None, description="9.1 แนบท้ายบันทึกเอกสารเลขอะไร")
    expense_items: List[ExpenseItem] = Field(default_factory=list, description="รายละเอียดพัสดุ/รายการ")
    total_amount: Optional[float] = Field(default=None, description="9.2 ยอดรวม")


class GeneralReceiptData(BaseModel):
    """10. ใบเสร็จรับเงิน/ใบกำกับภาษีทั่วไป"""
    vendor_name: Optional[str] = Field(default=None, description="ชื่อร้านค้าหรือบริษัทผู้ขาย")
    vendor_tax_id: Optional[str] = Field(default=None, description="เลขประจำตัวผู้เสียภาษี 13 หลัก")
    vendor_branch: Optional[str] = Field(default=None, description="สาขา")
    vendor_address: Optional[str] = Field(default=None, description="ที่อยู่ร้านค้า")
    customer_name: Optional[str] = Field(default=None, description="ชื่อผู้ซื้อ")
    customer_tax_id: Optional[str] = Field(default=None, description="เลขประจำตัวผู้ซื้อ")
    invoice_no: Optional[str] = Field(default=None, description="เลขที่ใบเสร็จ")
    doc_date: Optional[str] = Field(default=None, description="วันที่บนเอกสาร")
    line_items: List[ExpenseItem] = Field(default_factory=list, description="รายการสินค้า")
    subtotal: Optional[float] = Field(default=None, description="รวมเป็นเงินก่อนภาษี")
    vat: Optional[float] = Field(default=None, description="ภาษีมูลค่าเพิ่ม 7%")
    total_amount: Optional[float] = Field(default=None, description="ยอดเงินรวมทั้งสิ้น")



class ValidationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationIssue(BaseModel):
    """An issue identified by Component 4 during validation."""
    field: str = Field(description="ชื่อฟิลด์ที่มีปัญหา เช่น vendor_tax_id, total_amount")
    message: str = Field(description="คำอธิบายข้อผิดพลาดหรือข้อสังเกต")
    severity: ValidationSeverity = Field(default=ValidationSeverity.ERROR, description="ระดับความรุนแรง (ERROR/WARNING/INFO)")
    code: str = Field(description="รหัสข้อผิดพลาด เช่น MISSING_REQUIRED_FIELD, TAX_ID_INVALID")
    expected: Optional[Any] = Field(default=None, description="ค่าที่คาดหวัง")
    actual: Optional[Any] = Field(default=None, description="ค่าที่ตรวจพบจริง")


class MathReconciliationReport(BaseModel):
    """Financial arithmetic cross-validation report."""
    is_balanced: bool = Field(description="ยอดเงินสอดคล้องกันถูกต้องหรือไม่")
    expected_total: Optional[float] = Field(default=None, description="ยอดรวมที่คำนวณได้จากการตรวจสอบ")
    actual_total: Optional[float] = Field(default=None, description="ยอดรวมที่ระบุในเอกสาร")
    difference: Optional[float] = Field(default=0.0, description="ผลต่าง (Actual - Expected)")
    line_items_sum: Optional[float] = Field(default=None, description="ผลรวมของรายการสินค้า/ค่าใช้จ่ายแต่ละรายการ")
    details: str = Field(default="", description="รายละเอียดผลการกระทบยอด")


class TaxIdValidationReport(BaseModel):
    """Thai 13-digit Tax ID validation & formatting report."""
    raw_id: Optional[str] = Field(default=None, description="เลขประจำตัวผู้เสียภาษีดิบก่อนตรวจสอบ")
    cleaned_id: Optional[str] = Field(default=None, description="เลข 13 หลักที่ล้างอักขระพิเศษแล้ว")
    formatted_id: Optional[str] = Field(default=None, description="รูปแบบทางการ X-XXXX-XXXXX-XX-X")
    is_valid: bool = Field(default=False, description="ผ่านการตรวจสูตร Mod 11 กรมสรรพากรหรือไม่")
    error_message: Optional[str] = Field(default=None, description="สาเหตุหากไม่ผ่าน")


class DateNormalizationReport(BaseModel):
    """Thai / Christian Era date normalization report."""
    raw_date: Optional[str] = Field(default=None, description="ข้อความวันที่ดิบที่พบบนเอกสาร")
    iso_date: Optional[str] = Field(default=None, description="วันที่มาตรฐานสากล ISO 8601 (YYYY-MM-DD)")
    thai_formatted: Optional[str] = Field(default=None, description="วันที่ภาษาไทยเต็ม เช่น 18 กันยายน 2567")
    is_buddhist_era: bool = Field(default=False, description="ระบุว่าเป็นปี พ.ศ. หรือไม่")
    is_valid: bool = Field(default=False, description="สามารถแปลงวันที่ได้สำเร็จหรือไม่")


class DocumentValidationResult(BaseModel):
    """Component 4 complete validation and normalization output."""
    is_valid: bool = Field(description="ผ่านการตรวจสอบหลักหรือไม่ (ไม่มีข้อผิดพลาดระดับ ERROR)")
    status: str = Field(description="สถานะโดยรวม: PASSED, WARNING, ERROR")
    issues: List[ValidationIssue] = Field(default_factory=list, description="รายการข้อผิดพลาดหรือข้อสังเกตทั้งหมด")
    math_report: Optional[MathReconciliationReport] = Field(default=None, description="รายงานการตรวจสอบตัวเลขทางการเงิน")
    tax_id_report: Optional[TaxIdValidationReport] = Field(default=None, description="รายงานตรวจสอบเลขประจำตัวผู้เสียภาษี")
    date_report: Optional[DateNormalizationReport] = Field(default=None, description="รายงานการปรับรูปแบบวันที่")
    normalized_data: Dict[str, Any] = Field(default_factory=dict, description="ข้อมูลที่ปรับรูปแบบแล้ว (ISO Date, Formatted Tax ID, ฯลฯ)")


class TypeDirectedExtractionResult(BaseModel):
    """Unified result container returned to the UI and API."""
    document_type: str = Field(description="รหัสประเภทเอกสาร เช่น principle_approval_request")
    document_type_name_th: str = Field(description="ชื่อประเภทเอกสารภาษาไทย")
    
    # Generic dictionary of all key-value fields for flexible UI rendering
    fields: Dict[str, Any] = Field(default_factory=dict, description="พจนานุกรมฟิลด์ทั้งหมดที่สกัดได้")
    
    # Common structured lists/totals for consistent rendering
    expense_items: List[ExpenseItem] = Field(default_factory=list, description="รายการค่าใช้จ่าย/สินค้า")
    line_items: List[ExpenseItem] = Field(default_factory=list, description="Alias for expense_items")
    total_amount: Optional[float] = Field(default=None, description="ยอดรวมเงินหลักของเอกสาร")
    subtotal: Optional[float] = Field(default=None, description="ยอดรวมก่อนภาษี (ถ้ามี)")
    vat: Optional[float] = Field(default=None, description="ภาษีมูลค่าเพิ่ม (ถ้ามี)")

    # Backward compatibility fields for legacy receipt tests
    vendor_name: Optional[str] = Field(default=None, description="ชื่อผู้ขาย/ร้านค้า")
    vendor_tax_id: Optional[str] = Field(default=None, description="เลขประจำตัวผู้เสียภาษี")
    vendor_branch: Optional[str] = Field(default=None, description="สาขา")
    vendor_address: Optional[str] = Field(default=None, description="ที่อยู่")
    customer_name: Optional[str] = Field(default=None, description="ผู้ซื้อ")
    invoice_no: Optional[str] = Field(default=None, description="เลขที่เอกสาร")
    document_date: Optional[str] = Field(default=None, description="วันที่")
    
    # Component 4: Validation & Normalization Result
    validation: Optional[DocumentValidationResult] = Field(default=None, description="ผลการตรวจสอบความถูกต้องและจัดรูปแบบข้อมูล")

    # Metadata and audit trail
    thinking_process: Optional[str] = Field(default=None, description="ลำดับความคิดวิเคราะห์ (<think>)")
    model_used: Optional[str] = Field(default=None, description="ชื่อโมเดลที่ใช้")
    latency_ms: Optional[float] = Field(default=None, description="ความเร็วในการประมวลผล (ms)")
    is_mock: bool = Field(default=False, description="ระบุว่ามาจาก Mock หรือไม่")
    raw_llm_response: Optional[str] = Field(default=None, description="ข้อความ JSON ดิบจาก LLM")


# Backward compatible FinancialExtractionResult
class FinancialExtractionResult(BaseModel):
    """Backward compatible model matching previous API calls."""
    document_type: str = "ใบเสร็จรับเงิน"
    invoice_no: Optional[str] = None
    document_date: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_tax_id: Optional[str] = None
    vendor_branch: Optional[str] = None
    vendor_address: Optional[str] = None
    customer_name: Optional[str] = None
    customer_tax_id: Optional[str] = None
    line_items: List[ExpenseItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    vat: Optional[float] = None
    total_amount: Optional[float] = None
    thinking_process: Optional[str] = None
    model_used: Optional[str] = None
    latency_ms: Optional[float] = None
    is_mock: bool = False
    raw_llm_response: Optional[str] = None
    validation: Optional[DocumentValidationResult] = None


# =============================================================================
# Component 5: End-to-End Pipeline, REST API & Benchmarking Schemas
# =============================================================================

class PipelineTiming(BaseModel):
    """Detailed stage latencies for the End-to-End pipeline (in milliseconds)."""
    ingestion_ms: float = Field(default=0.0, description="เวลาแปลงไฟล์และเตรียมภาพ (Component 1)")
    ocr_ms: float = Field(default=0.0, description="เวลาตรวจจับและอ่านตัวอักษร OCR (Component 2)")
    llm_ms: float = Field(default=0.0, description="เวลาวิเคราะห์และสกัดข้อมูลด้วย LLM (Component 3)")
    validation_ms: float = Field(default=0.0, description="เวลาตรวจสอบและจัดรูปแบบข้อมูล (Component 4)")
    total_ms: float = Field(default=0.0, description="เวลารวมทั้งหมดของ Pipeline")


class FullPipelineResult(BaseModel):
    """Complete production output returned by the End-to-End Extraction Pipeline."""
    filename: str = Field(description="ชื่อไฟล์เอกสารต้นทาง")
    document_type: str = Field(description="รหัสประเภทเอกสาร เช่น principle_approval_request")
    document_type_name_th: str = Field(description="ชื่อประเภทเอกสารภาษาไทย")
    pages_count: int = Field(default=1, description="จำนวนหน้าทั้งหมดที่ตรวจพบในเอกสาร")
    timings: PipelineTiming = Field(description="สถิติความเร็วการประมวลผลแยกตามขั้นตอน")
    
    # Core extraction and validation payload
    extraction: TypeDirectedExtractionResult = Field(description="ข้อมูลผลลัพธ์ที่สกัดได้จาก Component 3")
    validation: DocumentValidationResult = Field(description="ผลการตรวจสอบและจัดรูปแบบข้อมูลจาก Component 4")
    
    # Perception metadata for visual overlays and audits
    raw_ocr_markdown: Optional[str] = Field(default=None, description="ข้อความ OCR ที่จัดรูปแบบสำหรับส่งให้ LLM")
    total_text_blocks: int = Field(default=0, description="จำนวนกรอบข้อความทั้งหมดที่อ่านได้")
    image_preview_base64: Optional[str] = Field(default=None, description="ภาพตัวอย่าง Base64 สำหรับการแสดงผลหน้าเว็บ")


class ModelBenchmarkResult(BaseModel):
    """Performance & accuracy metrics for a specific model evaluation run."""
    model_name: str = Field(description="ชื่อโมเดล เช่น qwen2.5:3b")
    latency_ms: float = Field(description="ความเร็วในการประมวลผล (มิลลิวินาที)")
    is_mock: bool = Field(default=False, description="ระบุว่าเป็นการรันจริงหรือ Mock Snapshot")
    extracted_field_count: int = Field(default=0, description="จำนวนฟิลด์ที่สกัดข้อมูลได้สำเร็จ")
    validation_status: str = Field(description="สถานะการตรวจสอบ: PASSED, WARNING, ERROR")
    is_valid: bool = Field(description="ผ่านการตรวจสอบหลักหรือไม่")
    total_amount_extracted: Optional[float] = Field(default=None, description="ยอดรวมเงินที่โมเดลสกัดได้")
    math_balanced: bool = Field(default=False, description="ตัวเลขการเงินสอดคล้องกันหรือไม่")
    has_thinking_trace: bool = Field(default=False, description="มี Chain-of-Thought (<think>) หรือไม่")


class BenchmarkRequest(BaseModel):
    """Payload for triggering comparative model evaluation."""
    document_type: str = Field(default="general_receipt", description="ประเภทเอกสารที่จะทดสอบ")
    models: List[str] = Field(default_factory=lambda: ["qwen2.5:3b"], description="รายชื่อโมเดลที่ต้องการเปรียบเทียบ")
    temperature: float = Field(default=0.0, description="อุณหภูมิที่ใช้ในการทดสอบ")
    force_mock: bool = Field(default=False, description="บังคับใช้ Mock Snapshot เพื่อเปรียบเทียบความเร็วพื้นฐาน")


class BenchmarkReport(BaseModel):
    """Consolidated benchmark report comparing multiple models for seminar presentation."""
    timestamp: str = Field(description="วันเวลาที่ทำการทดสอบ")
    sample_name: str = Field(description="ชื่อเอกสารตัวอย่างที่ใช้ทดสอบ")
    document_type: str = Field(description="ประเภทเอกสารที่ใช้ทดสอบ")
    results: List[ModelBenchmarkResult] = Field(default_factory=list, description="ผลการทดสอบแยกตามโมเดล")
    fastest_model: Optional[str] = Field(default=None, description="โมเดลที่ประมวลผลเร็วที่สุด")
    recommended_model: Optional[str] = Field(default=None, description="โมเดลที่แนะนำสำหรับการใช้งานจริง")
    markdown_table: Optional[str] = Field(default=None, description="ตารางเปรียบเทียบในรูปแบบ Markdown สำหรับใส่รายงาน")


class HealthCheckResponse(BaseModel):
    """System status and readiness of all 5 components."""
    status: str = Field(default="HEALTHY", description="สถานะภาพรวมของระบบ")
    version: str = Field(default="1.0.0", description="เวอร์ชันของ Information Extraction Pipeline")
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="สถานะความพร้อมของแต่ละโมดูลย่อย")
    timestamp: str = Field(description="เวลาปัจจุบันของระบบ")
