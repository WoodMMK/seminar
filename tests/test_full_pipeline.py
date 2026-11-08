"""Unit tests for orchestration without loading Paddle models, ResNet or Ollama."""

from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.full_pipeline import (
    CombinedValidationExtractionLLM,
    FullDocumentPipeline,
    combined_output_schema,
)


class FakeIngestor:
    def load_document(self, _file_bytes):
        return [SimpleNamespace(image=Image.new("RGB", (100, 100), "white"))]


class FakeClassifier:
    def predict_image(self, _image):
        return {
            "prediction": "07_02Inspectionform",
            "confidence": 0.91,
            "top_three": [
                {"class_name": "07_02Inspectionform", "probability": 0.91},
                {"class_name": "07Inspectionform", "probability": 0.07},
                {"class_name": "02Approved", "probability": 0.02},
            ],
            "probabilities": {},
            "checkpoint": "fake.pt",
            "device": "cpu",
        }


class FakePerception:
    text_blocks = [SimpleNamespace(confidence=0.95)]

    def to_llm_markdown(self):
        return "เลขที่ บพ. 88/2567\nลงชื่อ สมชาย ใจดี"


class FakeOcrEngine:
    def process_image(self, _image, page_number=1):
        assert page_number == 1
        return FakePerception()


class FakeCombinedLlm:
    model = "fake-qwen"

    def __init__(self, decision="MATCH"):
        self.decision = decision
        self.call_count = 0

    def validate_and_extract(self, **_kwargs):
        self.call_count += 1
        fields = (
            {"doc_no": "บพ. 88/2567", "signer_names": ["สมชาย ใจดี"]}
            if self.decision == "MATCH"
            else {}
        )
        return (
            {
                "decision": self.decision,
                "evidence": ["test evidence"],
                "ocr_quality": "GOOD",
                "extracted_fields": fields,
                "raw_content": "{}",
            },
            {"latency_ms": 1.0, "request_count": 1},
        )


class FakeReferenceStore:
    def get(self, predicted_class):
        return {
            "text": "ตัวอย่างใบตรวจรับพัสดุหน้าที่ 2",
            "average_confidence": 0.96,
            "line_count": 1,
            "source": f"fake/{predicted_class}/01.json",
        }


def build_pipeline(llm):
    return FullDocumentPipeline(
        classifier=FakeClassifier(),
        ingestor=FakeIngestor(),
        ocr_engine=FakeOcrEngine(),
        combined_llm=llm,
        reference_store=FakeReferenceStore(),
    )


def test_match_uses_one_llm_request_and_extracts():
    llm = FakeCombinedLlm("MATCH")
    result = build_pipeline(llm).process(b"document", "test.pdf")

    assert llm.call_count == 1
    assert result["timings"]["llm_request_count"] == 1
    assert result["final_document_type"] == "07_02Inspectionform"
    assert result["should_extract"] is True
    assert result["extraction"]["fields"]["doc_no"] == "บพ. 88/2567"
    assert result["validation"]["status"] == "PASSED"
    assert result["llm"]["reference_ocr_source"].endswith(
        "07_02Inspectionform/01.json"
    )


def test_additional_uses_one_llm_request_and_does_not_extract():
    llm = FakeCombinedLlm("ADDITIONAL")
    result = build_pipeline(llm).process(b"document", "test.pdf")

    assert llm.call_count == 1
    assert result["final_document_type"] == "10Additional_documents"
    assert result["should_extract"] is False
    assert result["extraction"] is None


def test_prompt_policy_does_not_force_poor_ocr_to_additional():
    import inspect

    source = inspect.getsource(
        CombinedValidationExtractionLLM.validate_and_extract
    )
    assert "ไม่ได้หมายความว่าเป็น ADDITIONAL โดยอัตโนมัติ" in source
    assert "ห้ามสร้างสถานะ UNCERTAIN" in source


def test_combined_schema_caps_extraction_arrays_and_strings():
    schema = combined_output_schema(
        {
            "expense_items": [{"description": "รายการ"}],
            "requester": "ชื่อผู้เบิก",
        }
    )
    fields = schema["properties"]["extracted_fields"]["properties"]
    assert fields["expense_items"]["maxItems"] == 50
    description = fields["expense_items"]["items"]["properties"]["description"]
    assert description["anyOf"][0]["maxLength"] == 500
    assert fields["requester"]["anyOf"][0]["maxLength"] == 500


if __name__ == "__main__":
    test_match_uses_one_llm_request_and_extracts()
    test_additional_uses_one_llm_request_and_does_not_extract()
    test_prompt_policy_does_not_force_poor_ocr_to_additional()
    test_combined_schema_caps_extraction_arrays_and_strings()
    print("Full pipeline tests passed.")
