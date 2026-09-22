"""
PP-ChatOCRv4 Engine Wrapper with Thai OCR and Knowledge Payload Generator.

Integrates PaddleX PP-ChatOCRv4-doc pipeline with:
- Thai OCR: th_PP-OCRv5_mobile_rec
- Layout Detection: PicoDet-S_layout_3cls
- Local LLM: Ollama (OpenAI-compatible endpoint http://localhost:11434/v1)
- Knowledge Payload generator for Vector DB (embed_text + filter_metadata)
"""

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Enforce OneDNN / PIR settings before paddle is loaded
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

import paddle
paddle.set_flags({'FLAGS_use_mkldnn': False})

from paddlex import create_pipeline
from paddlex.inference.pipelines import load_pipeline_config
from paddlex.inference.models.runners.paddle_static.config.pp_option import PaddlePredictorOption

from src.validator import FinancialDocumentValidator


class PPChatOCREngine:
    """
    Production-ready wrapper around PaddleX PP-ChatOCRv4.
    Provides visual document analysis, interactive question answering,
    and automatic formatting into dual Knowledge Payloads for Vector DBs.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/v1",
        llm_model: str = "qwen2.5:3b",
        api_key: str = "ollama",
    ):
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.api_key = api_key
        self._pipeline = None
        self._visual_cache: Dict[str, Dict[str, Any]] = {}
        self._validator = FinancialDocumentValidator()

    def get_pipeline(self):
        """Lazy initialization of PP-ChatOCRv4 pipeline."""
        if self._pipeline is None:
            cfg = load_pipeline_config('PP-ChatOCRv4-doc')
            cfg['use_mllm_predict'] = False
            
            # Use PicoDet-S_layout_3cls for maximum stability on CPU
            cfg['SubPipelines']['LayoutParser']['SubModules']['LayoutDetection']['model_name'] = 'PicoDet-S_layout_3cls'
            
            # Thai OCR recognition model
            cfg['SubPipelines']['LayoutParser']['SubPipelines']['GeneralOCR']['SubModules']['TextRecognition']['model_name'] = 'th_PP-OCRv5_mobile_rec'
            
            # Local Ollama LLM endpoint
            cfg['SubModules']['LLM_Chat']['base_url'] = self.ollama_url
            cfg['SubModules']['LLM_Chat']['model_name'] = self.llm_model
            cfg['SubModules']['LLM_Chat']['api_key'] = self.api_key

            pp_opt = PaddlePredictorOption()
            pp_opt.enable_new_ir = False

            self._pipeline = create_pipeline(config=cfg, device="cpu", pp_option=pp_opt)
        return self._pipeline

    def visual_predict(self, image_path: Union[str, Path], use_cache: bool = True) -> Dict[str, Any]:
        """
        Run layout analysis and OCR perception on the document image.
        Caches results by image path to enable instant multi-turn chatting.
        """
        img_str = str(Path(image_path).resolve())
        if use_cache and img_str in self._visual_cache:
            return self._visual_cache[img_str]

        pipeline = self.get_pipeline()
        raw_res_list = list(pipeline.visual_predict(img_str))
        if not raw_res_list:
            raise RuntimeError(f"Visual prediction returned empty results for {image_path}")

        first_res = raw_res_list[0]
        # In PaddleX 3.7.2, visual_info is nested inside visual_res['visual_info']
        sub_vi = first_res.get('visual_info', first_res)

        self._visual_cache[img_str] = {
            "sub_vi": sub_vi,
            "raw_res": first_res,
            "cached_at": time.time(),
        }
        return self._visual_cache[img_str]

    def chat_and_extract(
        self,
        image_path: Union[str, Path],
        questions: List[str],
        document_type: str = "receipt",
        custom_prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract answers to questions using PP-ChatOCRv4 and generate Knowledge Payload.
        """
        start_time = time.time()
        img_path = Path(image_path).resolve()

        # Step 1: Visual Perception
        visual_data = self.visual_predict(img_path, use_cache=use_cache)
        sub_vi = visual_data["sub_vi"]

        # Step 2: System Prompt Instruction
        task_desc = custom_prompt or (
            "สกัดข้อมูลสำคัญตามที่ระบุในรายการคำถาม โดยอิงจากข้อความในเอกสารอย่างเคร่งครัด "
            "หากไม่มีข้อมูลระบุชัดเจน ให้ตอบว่า 'ไม่มีระบุ'"
        )

        pipeline = self.get_pipeline()
        
        # Step 3: Run LLM Chat Extraction
        chat_raw = pipeline.chat(
            key_list=questions,
            visual_info=sub_vi,
            use_vector_retrieval=False,
            text_task_description=task_desc
        )
        
        chat_answers = chat_raw.get("chat_res", {})

        # Step 4: Extract OCR text summary for embedding context
        ocr_text_lines = []
        normal_texts = sub_vi.get("normal_text_dict", {})
        if isinstance(normal_texts, dict):
            for block in normal_texts.values():
                if isinstance(block, list):
                    ocr_text_lines.extend(str(item) for item in block)
                elif isinstance(block, str):
                    ocr_text_lines.append(block)

        # Step 5: Generate Knowledge Payload
        knowledge_payload = self._build_knowledge_payload(
            chat_answers=chat_answers,
            document_type=document_type,
            source_filename=img_path.name,
            ocr_text_lines=ocr_text_lines
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "source_file": img_path.name,
            "document_type": document_type,
            "questions": questions,
            "chat_answers": chat_answers,
            "knowledge_payload": knowledge_payload,
            "ocr_text_snippet": " ".join(ocr_text_lines[:20]),
            "latency_ms": latency_ms,
            "model_info": {
                "engine": "PaddleX PP-ChatOCRv4-doc",
                "layout_model": "PicoDet-S_layout_3cls",
                "ocr_model": "th_PP-OCRv5_mobile_rec",
                "llm_model": self.llm_model,
                "base_url": self.ollama_url,
            }
        }

    def _build_knowledge_payload(
        self,
        chat_answers: Dict[str, Any],
        document_type: str,
        source_filename: str,
        ocr_text_lines: List[str]
    ) -> Dict[str, Any]:
        """
        Builds the dual Knowledge Payload:
        1. embed_text: Dense Markdown representation designed for Vector DB embedding.
        2. filter_metadata: Exact key-value dictionary for metadata filtering.
        """
        # Heuristic extraction of common fields from answers
        vendor_name = None
        vendor_tax_id = None
        doc_date_str = None
        total_amount = None
        currency = "THB"

        for key, val in chat_answers.items():
            if not isinstance(val, str) or val in ("ไม่มีระบุ", "未知", "null", ""):
                continue
            
            k_lower = key.lower()
            if any(term in k_lower for term in ["ร้านค้า", "บริษัท", "ผู้ขาย", "ผู้ให้บริการ", "vendor"]):
                if not vendor_name:
                    vendor_name = val
            elif any(term in k_lower for term in ["เลขประจำตัวผู้เสียภาษี", "tax id", "tax_id"]):
                tax_match = re.search(r"\d{13}", val.replace("-", "").replace(" ", ""))
                if tax_match:
                    vendor_tax_id = tax_match.group(0)
                else:
                    vendor_tax_id = val
            elif any(term in k_lower for term in ["วันที่", "date"]):
                if not doc_date_str:
                    doc_date_str = val
            elif any(term in k_lower for term in ["ยอดเงิน", "ยอดรวม", "จำนวนเงิน", "total", "amount", "สุทธิ"]):
                # Parse numeric amount
                amt_match = re.search(r"[\d,]+(?:\.\d{1,2})?", val)
                if amt_match:
                    try:
                        clean_num = amt_match.group(0).replace(",", "")
                        total_amount = float(clean_num)
                    except ValueError:
                        pass

        # Normalize date to ISO YYYY-MM-DD if possible
        doc_date_iso = None
        if doc_date_str:
            norm_res = self._validator.normalize_thai_date(doc_date_str)
            if norm_res and norm_res.is_valid and norm_res.iso_date:
                doc_date_iso = norm_res.iso_date

        # Build Clean Embed Text
        doc_type_th_map = {
            "receipt": "ใบเสร็จรับเงิน (Receipt)",
            "tax_invoice": "ใบกำกับภาษี (Tax Invoice)",
            "invoice": "ใบแจ้งหนี้ (Invoice)",
            "payment_voucher": "ใบสำคัญรับเงิน (Payment Voucher)",
            "credit_card_slip": "สลิปบัตรเครดิต (Credit Card Slip)",
            "general_document": "เอกสารการเงินทั่วไป",
        }
        doc_type_display = doc_type_th_map.get(document_type, document_type)

        md_lines = [
            f"# เอกสารการเงิน: {doc_type_display}",
            f"- **ไฟล์ต้นฉบับ:** {source_filename}",
            f"- **ชื่อร้านค้า / ผู้ออกเอกสาร:** {vendor_name or 'ไม่ระบุ'}",
            f"- **วันที่เอกสาร:** {doc_date_str or 'ไม่ระบุ'}" + (f" (ISO: {doc_date_iso})" if doc_date_iso else ""),
            f"- **ยอดเงินรวมทั้งสิ้น:** {f'{total_amount:,.2f} บาท' if total_amount is not None else 'ไม่ระบุ'}",
            f"- **เลขประจำตัวผู้เสียภาษี:** {vendor_tax_id or 'ไม่ระบุ'}",
            "",
            "## ข้อมูลที่สกัดได้ตามคำถาม (PP-ChatOCRv4):",
        ]

        for q, a in chat_answers.items():
            md_lines.append(f"- **{q}:** {a}")

        embed_text = "\n".join(md_lines)

        filter_metadata = {
            "document_type": document_type,
            "vendor_name": vendor_name,
            "vendor_tax_id": vendor_tax_id,
            "doc_date_iso": doc_date_iso,
            "total_amount": total_amount,
            "currency": currency,
            "source_filename": source_filename,
            "extracted_at": datetime.now().isoformat(),
            "extracted_by": f"PP-ChatOCRv4 ({self.llm_model})"
        }

        return {
            "embed_text": embed_text,
            "filter_metadata": filter_metadata
        }
