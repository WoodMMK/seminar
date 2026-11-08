"""Focused tests for the latest ResNet-to-extraction contract."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.llm_extractor import LLMExtractor, PROMPT_CONFIGS
from src.full_pipeline import ReferenceOcrStore
from src.schemas import (
    DocumentType,
    RESNET_CLASS_TO_DOCUMENT_TYPE,
    resolve_document_type,
)


def test_resnet_mapping_keeps_form_1_and_form_2_correct():
    assert resolve_document_type("04PettyCash") == DocumentType.ADVANCE_PAYMENT_REQUEST_2.value
    assert resolve_document_type("05AdvancePayment") == DocumentType.ADVANCE_PAYMENT_REQUEST_1.value


def test_latest_resnet_taxonomy_is_complete():
    expected = {
        "01Request", "02Approved", "03RequestReim", "04PettyCash",
        "05AdvancePayment", "06CertificateInsteadofReceipt",
        "07Inspectionform", "07_02Inspectionform", "08RequestMaterial",
        "09Attchedmaterial", "10Approved_materials",
        "10Additional_documents",
    }
    assert set(RESNET_CLASS_TO_DOCUMENT_TYPE) == expected


def test_new_prompts_contain_the_requested_fields():
    page_2 = PROMPT_CONFIGS[DocumentType.PARCEL_INSPECTION_PAGE_2.value]["json_schema"]
    assert set(page_2) == {"doc_no", "signer_names"}

    approved = PROMPT_CONFIGS[DocumentType.APPROVED_MATERIALS.value]["json_schema"]
    assert set(approved) == {
        "procurement_request_code",
        "approved_total_amount",
        "approved_items",
        "document_date",
        "title",
        "parcel_inspectors",
    }


def test_additional_document_is_skipped_without_extraction():
    result = LLMExtractor().extract(
        ocr_markdown="ใบเสร็จทั่วไป",
        document_type="10Additional_documents",
        force_mock=True,
    )
    assert result.document_type == DocumentType.ADDITIONAL_DOCUMENTS.value
    assert result.should_extract is False
    assert result.fields == {}
    assert result.validation is None
    assert result.model_used is None


def test_new_types_build_structured_mock_results():
    extractor = LLMExtractor()

    page_2 = extractor.extract("OCR", "07_02Inspectionform", force_mock=True)
    assert page_2.document_type == DocumentType.PARCEL_INSPECTION_PAGE_2.value
    assert page_2.fields["doc_no"]
    assert page_2.fields["signer_names"]
    assert page_2.validation is not None
    assert page_2.validation.status == "PASSED"

    approved = extractor.extract("OCR", "10Approved_materials", force_mock=True)
    assert approved.document_type == DocumentType.APPROVED_MATERIALS.value
    assert approved.fields["procurement_request_code"]
    assert approved.total_amount == approved.fields["approved_total_amount"]
    assert approved.expense_items
    assert approved.validation is not None
    assert approved.validation.status == "PASSED"


def test_reference_ocr_is_available_for_every_extractable_resnet_class():
    store = ReferenceOcrStore()
    for class_name in RESNET_CLASS_TO_DOCUMENT_TYPE:
        if class_name == "10Additional_documents":
            continue
        reference = store.get(class_name)
        assert reference["text"].strip()
        assert reference["source"].endswith(f"{class_name}\\01.json")


if __name__ == "__main__":
    test_resnet_mapping_keeps_form_1_and_form_2_correct()
    test_latest_resnet_taxonomy_is_complete()
    test_new_prompts_contain_the_requested_fields()
    test_additional_document_is_skipped_without_extraction()
    test_new_types_build_structured_mock_results()
    test_reference_ocr_is_available_for_every_extractable_resnet_class()
    print("Latest taxonomy tests passed.")
