"""
Component 5: Model Benchmarking & Performance Evaluation Engine
Evaluates and compares local LLM extraction models (latency, field completeness,
validation pass rate, math accuracy) for seminar presentation and academic thesis.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm_extractor import LLMExtractor
from src.validator import FinancialDocumentValidator
from src.schemas import (
    BenchmarkReport,
    DocumentType,
    ModelBenchmarkResult,
)

SAMPLE_OCR_BENCHMARK = """--- Page 1 ---
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


class ModelBenchmarker:
    """
    Evaluates and benchmarks multiple LLM models on financial information extraction tasks.
    Generates comparison metrics and publication-ready Markdown tables.
    """

    def __init__(
        self,
        extractor: Optional[LLMExtractor] = None,
        validator: Optional[FinancialDocumentValidator] = None,
    ):
        self.extractor = extractor or LLMExtractor()
        self.validator = validator or FinancialDocumentValidator()

    def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        ocr_markdown: str = SAMPLE_OCR_BENCHMARK,
        document_type: str = DocumentType.GENERAL_RECEIPT.value,
        sample_name: str = "sample_receipt_stationery",
        temperature: float = 0.0,
        include_mock_baseline: bool = True,
    ) -> BenchmarkReport:
        """
        Executes benchmark comparison across specified models.
        """
        available_models = self.extractor.list_available_models() if self.extractor.is_ollama_running() else []
        test_models = list(models) if models else []

        if not test_models:
            if available_models:
                test_models = available_models
            else:
                test_models = ["qwen2.5:3b"]

        results: List[ModelBenchmarkResult] = []

        # 1. Baseline Benchmark: Snapshot Mock
        if include_mock_baseline:
            t0 = time.perf_counter()
            mock_res = self.extractor.extract(
                ocr_markdown=ocr_markdown,
                document_type=document_type,
                force_mock=True,
            )
            mock_val = self.validator.validate(mock_res)
            latency = mock_res.latency_ms or round((time.perf_counter() - t0) * 1000, 2)

            results.append(
                ModelBenchmarkResult(
                    model_name=f"{mock_res.model_used}",
                    latency_ms=latency,
                    is_mock=True,
                    extracted_field_count=len(mock_res.fields or {}),
                    validation_status=mock_val.status,
                    is_valid=mock_val.is_valid,
                    total_amount_extracted=mock_res.total_amount,
                    math_balanced=mock_val.math_report.is_balanced if mock_val.math_report else False,
                    has_thinking_trace=bool(mock_res.thinking_process),
                )
            )

        # 2. Live Local LLM Evaluation
        for model in test_models:
            t0 = time.perf_counter()
            try:
                live_res = self.extractor.extract(
                    ocr_markdown=ocr_markdown,
                    document_type=document_type,
                    model_name=model,
                    temperature=temperature,
                    force_mock=False,
                )
                live_val = self.validator.validate(live_res)
                latency = live_res.latency_ms or round((time.perf_counter() - t0) * 1000, 2)

                results.append(
                    ModelBenchmarkResult(
                        model_name=model,
                        latency_ms=latency,
                        is_mock=live_res.is_mock,
                        extracted_field_count=len(live_res.fields or {}),
                        validation_status=live_val.status,
                        is_valid=live_val.is_valid,
                        total_amount_extracted=live_res.total_amount,
                        math_balanced=live_val.math_report.is_balanced if live_val.math_report else False,
                        has_thinking_trace=bool(live_res.thinking_process),
                    )
                )
            except Exception as e:
                print(f"[ModelBenchmarker] Error testing model '{model}': {e}")
                results.append(
                    ModelBenchmarkResult(
                        model_name=model,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                        is_mock=False,
                        extracted_field_count=0,
                        validation_status="ERROR",
                        is_valid=False,
                        total_amount_extracted=None,
                        math_balanced=False,
                        has_thinking_trace=False,
                    )
                )

        # Determine fastest and recommended
        live_results = [r for r in results if not r.is_mock]
        if live_results:
            fastest = min(live_results, key=lambda r: r.latency_ms).model_name
            recommended = fastest
        else:
            fastest = min(results, key=lambda r: r.latency_ms).model_name if results else None
            recommended = fastest

        md_table = self._generate_markdown_table(results)

        return BenchmarkReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            sample_name=sample_name,
            document_type=document_type,
            results=results,
            fastest_model=fastest,
            recommended_model=recommended,
            markdown_table=md_table,
        )

    @staticmethod
    def _generate_markdown_table(results: List[ModelBenchmarkResult]) -> str:
        """Constructs a Markdown comparison table suitable for academic reports."""
        lines = [
            "| Model / Configuration | Latency (s) | Fields Extracted | Validation Status | Math Balanced | CoT Reasoning | Evaluation Note |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for r in results:
            sec_str = f"{(r.latency_ms / 1000):.2f}s"
            math_icon = "✓" if r.math_balanced else "❌"
            cot_icon = "✓ Yes" if r.has_thinking_trace else "- No"
            if r.is_mock:
                note = "Snapshot Mock Baseline"
            elif r.validation_status == "PASSED":
                note = "100% Passed (Zero Errors)"
            elif r.validation_status == "WARNING":
                note = "Passed with Audit Warnings"
            else:
                note = "Validation Errors Detected"

            lines.append(
                f"| **{r.model_name}** | `{sec_str}` | {r.extracted_field_count} fields | `{r.validation_status}` | {math_icon} | {cot_icon} | {note} |"
            )

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run LLM Model Extraction Benchmark")
    parser.add_argument("--doc-type", default="general_receipt", help="Document type to benchmark")
    parser.add_argument("--models", nargs="*", default=["qwen2.5:3b"], help="Model names to test")
    parser.add_argument("--no-mock", action="store_true", help="Skip snapshot mock baseline")
    parser.add_argument("--output", default=None, help="Save report to JSON file path")
    args = parser.parse_args()

    print("=" * 68)
    print("🚀 Running Financial Document Model Benchmark (Component 5)...")
    print("=" * 68)

    benchmarker = ModelBenchmarker()
    report = benchmarker.run_benchmark(
        models=args.models,
        document_type=args.doc_type,
        include_mock_baseline=not args.no_mock,
    )

    print("\n" + report.markdown_table + "\n")
    print(f"⚡ Fastest Model: {report.fastest_model}")
    print(f"🎯 Recommended Model: {report.recommended_model}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"💾 Report saved to: {out_path}")


if __name__ == "__main__":
    main()
