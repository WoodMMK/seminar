"""
PP-ChatOCRv4 Engine Wrapper with Thai OCR, Automated Templates & Knowledge Payload Generator.

Integrates PaddleX PP-ChatOCRv4-doc pipeline with:
- Thai OCR: th_PP-OCRv5_mobile_rec
- Layout Detection: PicoDet-S_layout_3cls
- Local LLM: Ollama (OpenAI-compatible endpoint http://localhost:11434/v1)
- Automated Type-Directed System Prompts and Target Key Lists for 10 document types
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


# =========================================================================
# Predefined Type-Directed Templates for 10 Document Types
# =========================================================================
PP_CHATOCR_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "principle_approval_request": {
        "title": "1. เอกสารขออนุมัติหลักการ",
        "keys": [
            "เลขที่เอกสารหรือเลขที่หนังสือ",
            "วันที่ทำเอกสาร",
            "เรื่อง",
            "ผู้ทำการเบิกหรือหน่วยงานที่ขอ",
            "รายละเอียดค่าใช้จ่าย",
            "ยอดรวมเงินงบประมาณที่ขออนุมัติ"
        ],
        "system_prompt": (
            "คุณเป็นผู้เชี่ยวชาญการสกัดข้อมูลเอกสารราชการและเอกสารการเงิน หน้าที่ของคุณคือสกัดข้อมูลจากบันทึกข้อความ 'ขออนุมัติหลักการ' "
            "โดยอ้างอิงจากข้อความในเอกสารอย่างเคร่งครัด สกัดเลขที่หนังสือ วันที่ เรื่อง ผู้ขออนุมัติ และยอดเงินรวมที่ขออนุมัติ "
            "หากไม่มีข้อมูลระบุชัดเจนให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "principle_approval_granted": {
        "title": "2. เอกสารอนุมัติหลักการ",
        "keys": [
            "ตามหนังสือขออนุมัติหลักการเลขที่",
            "รายละเอียดการอนุมัติ",
            "ยอดรวมเงินที่ได้รับการอนุมัติ"
        ],
        "system_prompt": (
            "สกัดข้อมูลจาก 'เอกสารอนุมัติหลักการ' โดยดึงเลขที่หนังสือเดิมที่อ้างอิงถึง รายการที่อนุมัติ และยอดเงินที่อนุมัติทั้งหมด "
            "หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "disbursement_approval_request": {
        "title": "3. ขออนุมัติเบิกจ่าย",
        "keys": [
            "เลขที่เอกสาร",
            "วันที่ทำเอกสาร",
            "เรื่อง",
            "รายละเอียดค่าใช้จ่าย",
            "ยอดรวมเงินที่ขอเบิก",
            "ประเภทของการเบิกจ่าย (เงินสดย่อย หรือ ทดรองจ่าย)"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากบันทึกข้อความ 'ขออนุมัติเบิกจ่าย' ระบุเลขที่ วันที่ เรื่อง รายละเอียดค่าใช้จ่าย ยอดรวม "
            "และระบุว่าเป็นการเบิกจ่ายประเภทใด (เช่น เงินสดย่อย หรือ ทดรองจ่าย) หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "advance_payment_request_1": {
        "title": "4. แบบเบิกเงินทดรองจ่าย (แบบที่ 1 - สัญญายืมเงิน/เบิก)",
        "keys": [
            "เลขที่เอกสาร",
            "ผู้ทำการเบิกหรือผู้ยืมเงิน",
            "ยอดเงินรวมที่ขอยืม",
            "ตามหนังสืออนุมัติเบิกจ่ายเลขที่",
            "รายละเอียดตามประเภทค่าใช้จ่าย",
            "โอนเงินไปที่ใด (ธนาคารหรือเลขบัญชี)"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากแบบเบิกเงินทดรองจ่าย (แบบที่ 1) ระบุชื่อผู้เบิก ยอดเงินยืม เลขที่หนังสืออ้างอิง "
            "และช่องทางการโอนเงินหรือบัญชีผู้รับ หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "advance_payment_request_2": {
        "title": "5. แบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์เงิน)",
        "keys": [
            "เลขที่เอกสาร",
            "วันที่ส่งเอกสาร",
            "วันที่ขอรับเงิน",
            "ผู้ทำการเบิก",
            "ยอดเงินรวม",
            "ตามหนังสืออนุมัติหลักการเลขที่",
            "รายละเอียดค่าใช้จ่าย",
            "โอนเงินไปที่ใด"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากแบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์เงิน) ระบุวันที่ส่ง วันที่ขอรับเงิน "
            "ผู้เบิก ยอดเงินรวม และบัญชีปลายทาง หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "receipt_substitute": {
        "title": "6. ใบแทนใบเสร็จ / ใบสำคัญรับเงิน",
        "keys": [
            "วันเดือนปีที่จ่ายเงิน",
            "รายละเอียดของรายการการเบิก",
            "ยอดเงินรวมทั้งสิ้น",
            "ผู้จ่ายเงินหรือผู้รับรองการจ่าย"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากใบแทนใบเสร็จรับเงิน หรือใบสำคัญรับเงิน ระบุวันเดือนปี รายการการเบิก ยอดเงินรวม "
            "และชื่อผู้จ่ายเงิน หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "parcel_inspection": {
        "title": "7. ใบตรวจรับพัสดุ",
        "keys": [
            "เลขที่เอกสาร",
            "รายชื่อผู้ตรวจรับพัสดุหรือคณะกรรมการ",
            "วันที่ตรวจรับพัสดุ",
            "ตามใบสั่งซื้อหรือสัญญาเลขที่"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากใบตรวจรับพัสดุ ระบุเลขที่เอกสาร รายชื่อคณะกรรมการตรวจรับพัสดุ วันที่ตรวจรับ "
            "และเลขที่ใบสั่งซื้อหรือสัญญา หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "procurement_approval_request": {
        "title": "8. ขออนุมัติจัดหาพัสดุ",
        "keys": [
            "เลขที่เอกสาร",
            "วันที่ทำเอกสาร",
            "เรื่อง",
            "เหตุผลความจำเป็นที่ต้องจัดหา",
            "รายละเอียดของพัสดุ",
            "วงเงินงบประมาณที่ใช้ทั้งหมด",
            "เวลาที่ต้องใช้พัสดุ"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากบันทึกขออนุมัติจัดหาพัสดุ ระบุเลขที่ วันที่ เรื่อง เหตุผล รายการพัสดุ "
            "วงเงินงบประมาณ และกำหนดเวลาที่ต้องใช้พัสดุ หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "procurement_attachment": {
        "title": "9. เอกสารประกอบการขออนุมัติจัดหา",
        "keys": [
            "แนบท้ายบันทึกเอกสารเลขที่",
            "รายละเอียดพัสดุหรือรายการเปรียบเทียบราคา",
            "ยอดรวมงบประมาณ"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากเอกสารประกอบการขออนุมัติจัดหา ระบุเลขที่แนบท้ายบันทึก รายละเอียดพัสดุ และยอดรวม "
            "หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "general_receipt": {
        "title": "10. ใบเสร็จรับเงิน / ใบกำกับภาษีทั่วไป",
        "keys": [
            "ชื่อร้านค้าหรือบริษัทผู้ขาย",
            "เลขประจำตัวผู้เสียภาษี 13 หลัก",
            "วันที่ออกเอกสาร",
            "ยอดรวมก่อนภาษี (Subtotal)",
            "ภาษีมูลค่าเพิ่ม 7% (VAT)",
            "ยอดเงินรวมทั้งสิ้น (Total Amount)",
            "รายการสินค้าและบริการ"
        ],
        "system_prompt": (
            "คุณเป็นผู้เชี่ยวชาญการตรวจเอกสารการเงิน สกัดข้อมูลจากใบเสร็จรับเงินหรือใบกำกับภาษีอย่างแม่นยำ "
            "ระบุชื่อร้านค้า เลขประจำตัวผู้เสียภาษี 13 หลัก วันที่ ยอดรวมก่อนภาษี VAT 7% และยอดสุทธิรวมทั้งสิ้น "
            "หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    }
}

# Aliases for flexible matching
DOCUMENT_TYPE_ALIASES = {
    "receipt": "general_receipt",
    "tax_invoice": "general_receipt",
    "invoice": "general_receipt",
    "payment_voucher": "receipt_substitute",
    "credit_card_slip": "general_receipt",
    "general_document": "general_receipt",
}


class PPChatOCREngine:
    """
    Production-ready wrapper around PaddleX PP-ChatOCRv4.
    Provides visual document analysis, type-directed automated question answering,
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

    @staticmethod
    def get_supported_templates() -> List[Dict[str, Any]]:
        """Returns metadata of all 10 supported document type templates."""
        results = []
        for key, conf in PP_CHATOCR_TEMPLATES.items():
            results.append({
                "id": key,
                "title": conf["title"],
                "keys": conf["keys"],
                "system_prompt": conf["system_prompt"]
            })
        return results

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
        questions: Optional[List[str]] = None,
        document_type: str = "general_receipt",
        custom_prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Automated extraction using PP-ChatOCRv4 with template system prompt & keys.
        """
        start_time = time.time()
        img_path = Path(image_path).resolve()

        # Resolve document type and retrieve template
        canonical_doc_type = DOCUMENT_TYPE_ALIASES.get(document_type, document_type)
        template = PP_CHATOCR_TEMPLATES.get(
            canonical_doc_type,
            PP_CHATOCR_TEMPLATES["general_receipt"]
        )

        # 100% Automated fallback: Use template's predefined keys and prompt
        effective_questions = questions if (questions and len(questions) > 0) else template["keys"]
        effective_prompt = custom_prompt if custom_prompt else template["system_prompt"]

        # Step 1: Visual Perception
        visual_data = self.visual_predict(img_path, use_cache=use_cache)
        sub_vi = visual_data["sub_vi"]

        pipeline = self.get_pipeline()
        
        # Step 2: Run LLM Chat Extraction
        chat_raw = pipeline.chat(
            key_list=effective_questions,
            visual_info=sub_vi,
            use_vector_retrieval=False,
            text_task_description=effective_prompt
        )
        
        chat_answers = chat_raw.get("chat_res", {})

        # Step 3: Extract OCR text lines for context
        ocr_text_lines = []
        normal_texts = sub_vi.get("normal_text_dict", {})
        if isinstance(normal_texts, dict):
            for block in normal_texts.values():
                if isinstance(block, list):
                    ocr_text_lines.extend(str(item) for item in block)
                elif isinstance(block, str):
                    ocr_text_lines.append(block)

        # Step 4: Generate Knowledge Payload
        knowledge_payload = self._build_knowledge_payload(
            chat_answers=chat_answers,
            document_type=canonical_doc_type,
            doc_title=template["title"],
            source_filename=img_path.name,
            ocr_text_lines=ocr_text_lines
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "source_file": img_path.name,
            "document_type": canonical_doc_type,
            "document_title": template["title"],
            "questions": effective_questions,
            "system_prompt_used": effective_prompt,
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
        doc_title: str,
        source_filename: str,
        ocr_text_lines: List[str]
    ) -> Dict[str, Any]:
        """
        Builds the dual Knowledge Payload:
        1. embed_text: Dense Markdown representation designed for Vector DB embedding.
        2. filter_metadata: Exact key-value dictionary for metadata filtering.
        """
        doc_no = None
        vendor_or_requester = None
        vendor_tax_id = None
        doc_date_str = None
        total_amount = None
        currency = "THB"

        for key, val in chat_answers.items():
            if not isinstance(val, str) or val in ("ไม่มีระบุ", "未知", "null", ""):
                continue
            
            k_lower = key.lower()

            # 1. Document No
            if any(term in k_lower for term in ["เลขที่", "doc_no", "no."]):
                if not doc_no:
                    doc_no = val

            # 2. Requester or Vendor
            if any(term in k_lower for term in ["ผู้ทำการเบิก", "ผู้ยืม", "ผู้ขอ", "ร้านค้า", "บริษัท", "ผู้ขาย", "requester", "vendor"]):
                if not vendor_or_requester:
                    vendor_or_requester = val

            # 3. Tax ID
            if any(term in k_lower for term in ["เลขประจำตัวผู้เสียภาษี", "tax id", "tax_id"]):
                tax_match = re.search(r"\d{13}", val.replace("-", "").replace(" ", ""))
                vendor_tax_id = tax_match.group(0) if tax_match else val

            # 4. Date
            if any(term in k_lower for term in ["วันที่", "date", "วันเดือนปี"]):
                if not doc_date_str:
                    doc_date_str = val

            # 5. Amount
            if any(term in k_lower for term in ["ยอดเงิน", "ยอดรวม", "จำนวนเงิน", "งบประมาณ", "total", "amount", "สุทธิ"]):
                amt_match = re.search(r"[\d,]+(?:\.\d{1,2})?", val)
                if amt_match:
                    try:
                        clean_num = amt_match.group(0).replace(",", "")
                        total_amount = float(clean_num)
                    except ValueError:
                        pass

        # Normalize date to ISO YYYY-MM-DD
        doc_date_iso = None
        if doc_date_str:
            norm_res = self._validator.normalize_thai_date(doc_date_str)
            if norm_res and norm_res.is_valid and norm_res.iso_date:
                doc_date_iso = norm_res.iso_date

        # Build Clean Embed Text
        md_lines = [
            f"# เอกสารการเงิน: {doc_title}",
            f"- **ไฟล์ต้นฉบับ:** {source_filename}",
            f"- **ประเภทเอกสาร:** {doc_title}",
            f"- **เลขที่เอกสาร:** {doc_no or 'ไม่ระบุ'}",
            f"- **บุคคล/หน่วยงาน/ร้านค้า:** {vendor_or_requester or 'ไม่ระบุ'}",
            f"- **วันที่เอกสาร:** {doc_date_str or 'ไม่ระบุ'}" + (f" (ISO: {doc_date_iso})" if doc_date_iso else ""),
            f"- **ยอดเงินรวม:** {f'{total_amount:,.2f} บาท' if total_amount is not None else 'ไม่ระบุ'}",
        ]
        if vendor_tax_id:
            md_lines.append(f"- **เลขประจำตัวผู้เสียภาษี:** {vendor_tax_id}")

        md_lines.append("")
        md_lines.append("## ข้อมูลที่สกัดได้ตาม Template (PP-ChatOCRv4):")
        for q, a in chat_answers.items():
            md_lines.append(f"- **{q}:** {a}")

        embed_text = "\n".join(md_lines)

        filter_metadata = {
            "document_type": document_type,
            "document_title": doc_title,
            "doc_no": doc_no,
            "vendor_or_requester": vendor_or_requester,
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
