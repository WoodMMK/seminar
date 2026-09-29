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
        "title": "เอกสารขออนุมัติหลักการ",
        "group": "official_reimbursement",
        "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
        "keys": [
            "เลขที่เอกสาร",
            "วันที่ทำเอกสาร",
            "เรื่อง",
            "ผู้ทำการเบิก (คน หรือ ภาควิชา)",
            "รายละเอียดค่าใช้จ่าย (เบิกอะไรบ้าง และเท่าไหร่บ้าง)",
            "ยอดรวม"
        ],
        "system_prompt": (
            "คุณเป็นผู้เชี่ยวชาญการสกัดข้อมูลเอกสารราชการ หน้าที่ของคุณคือสกัดข้อมูลจากบันทึกข้อความ 'ขออนุมัติหลักการ' "
            "โดยอ้างอิงจากข้อความในเอกสารอย่างเคร่งครัด สกัด: "
            "1. เลขที่เอกสาร (สังเกตข้อความระบุ 'ที่' หรือ 'ที่ อว' เช่น ที่ อว 78.101/...) "
            "2. วันที่ทำเอกสาร "
            "3. เรื่อง "
            "4. ผู้ทำการเบิก (อาจเป็นชื่อบุคคล หรือชื่อภาควิชา) "
            "5. รายละเอียดค่าใช้จ่าย (แจกแจงรายการว่าเบิกอะไรบ้าง และรายการละเท่าไหร่บ้าง) "
            "6. ยอดรวม "
            "หากไม่มีข้อมูลระบุชัดเจนให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "principle_approval_granted": {
        "title": "เอกสารอนุมัติหลักการ",
        "group": "official_reimbursement",
        "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
        "keys": [
            "รายละเอียดการอนุมัติ (อนุมัติอะไรบ้าง เท่าไหร่บ้าง)",
            "ยอดรวมที่อนุมัติ",
            "ตามหนังสือขออนุมัติหลักการเลขที่ (ถ้ามี)"
        ],
        "system_prompt": (
            "สกัดข้อมูลจาก 'เอกสารอนุมัติหลักการ' ระบุ: "
            "1. รายละเอียดการอนุมัติ (อนุมัติรายการอะไรบ้าง และเท่าไหร่บ้าง) "
            "2. ยอดรวมที่อนุมัติ "
            "3. ตามหนังสือขออนุมัติหลักการเลขที่เดิมที่อ้างอิงถึง หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "disbursement_approval_request": {
        "title": "ขออนุมัติเบิกจ่าย",
        "group": "official_reimbursement",
        "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
        "keys": [
            "เลขที่เอกสาร",
            "วันที่ทำเอกสาร",
            "เรื่อง",
            "รายละเอียดค่าใช้จ่าย (เบิกอะไรบ้าง และเท่าไหร่บ้าง)",
            "ยอดรวม",
            "ประเภทของการเบิกจ่าย (เงินสดย่อย / ทดรองจ่าย)",
            "อ้างถึงบันทึกหลักการเลขที่ (ถ้ามี)"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากบันทึกข้อความ 'ขออนุมัติเบิกจ่าย' ระบุ: "
            "1. เลขที่เอกสาร 2. วันที่ทำเอกสาร 3. เรื่อง 4. รายละเอียดค่าใช้จ่าย (เบิกอะไรบ้าง และเท่าไหร่บ้าง) "
            "5. ยอดรวม 6. ประเภทของการเบิกจ่าย (ระบุชัดเจนว่าเป็น เงินสดย่อย หรือ ทดรองจ่าย) "
            "พร้อมระบุเลขที่หลักการเดิมที่อ้างอิงถึง หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "advance_payment_request_1": {
        "title": "แบบเบิกเงินทดรองจ่าย (แบบที่ 1 - สัญญายืมเงิน/เบิก)",
        "group": "official_reimbursement",
        "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
        "keys": [
            "เลขที่เอกสาร",
            "ใครเป็นคนเบิก",
            "ยอดรวม",
            "ตามหนังสืออนุมัติเบิกจ่าย เลขที่อะไร",
            "รายละเอียด (ดึงข้อมูลตามประเภทค่าใช้จ่าย)",
            "โอนเงินไปที่ใด"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากแบบเบิกเงินทดรองจ่าย (แบบที่ 1) ระบุ: "
            "1. เลขที่เอกสาร 2. ใครเป็นคนเบิก 3. ยอดรวม 4. ตามหนังสืออนุมัติเบิกจ่าย เลขที่อะไร "
            "5. รายละเอียดค่าใช้จ่าย (แจกแจงตามประเภทของค่าใช้จ่าย) 6. โอนเงินไปที่ใด (ธนาคารหรือเลขที่บัญชี) "
            "หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "advance_payment_request_2": {
        "title": "แบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์เงิน)",
        "group": "official_reimbursement",
        "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
        "keys": [
            "เลขที่เอกสาร",
            "วันที่ส่งเอกสาร",
            "วันที่ขอรับเงิน",
            "ใครเป็นคนเบิก",
            "ยอดรวม",
            "ตามหนังสืออนุมัติหลักการ เลขที่อะไร",
            "รายละเอียด",
            "โอนเงินไปที่ใด"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากแบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์เงิน) ระบุ: "
            "1. เลขที่เอกสาร 2. วันที่ส่งเอกสาร 3. วันที่ขอรับเงิน 4. ใครเป็นคนเบิก 5. ยอดรวม "
            "6. ตามหนังสืออนุมัติหลักการ เลขที่อะไร 7. รายละเอียดค่าใช้จ่าย 8. โอนเงินไปที่ใด "
            "หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "receipt_substitute": {
        "title": "ใบแทนใบเสร็จ / ใบสำคัญรับเงิน",
        "group": "official_reimbursement",
        "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
        "keys": [
            "วัน/เดือน/ปี",
            "รายละเอียดของรายการการเบิก",
            "ยอดรวม",
            "ผู้จ่ายเงิน"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากใบแทนใบเสร็จรับเงิน หรือใบสำคัญรับเงิน ระบุ: "
            "1. วัน/เดือน/ปี 2. รายละเอียดของรายการการเบิก 3. ยอดรวม 4. ผู้จ่ายเงิน (ชื่อผู้จ่ายหรือผู้รับรอง) "
            "หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "parcel_inspection": {
        "title": "ใบตรวจรับพัสดุ",
        "group": "official_reimbursement",
        "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
        "keys": [
            "เลขที่เอกสาร",
            "ผู้ตรวจรับพัสดุ",
            "วันที่ตรวจรับพัสดุ",
            "ตามใบสั่งซื้อหรือสัญญาเลขที่"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากใบตรวจรับพัสดุ ระบุ: "
            "1. เลขที่เอกสาร 2. ผู้ตรวจรับพัสดุ (รายชื่อผู้ตรวจรับหรือคณะกรรมการ) 3. วันที่ตรวจรับพัสดุ 4. ตามใบสั่งซื้อหรือสัญญาเลขที่ "
            "หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "procurement_approval_request": {
        "title": "ขออนุมัติหาพัสดุ (ขออนุมัติจัดหาพัสดุ)",
        "group": "official_reimbursement",
        "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
        "keys": [
            "เลขที่เอกสาร",
            "วันที่ทำเอกสาร",
            "เรื่อง",
            "เหตุผลที่ต้องจัดหา",
            "รายละเอียดของพัสดุ",
            "วงเงินที่ใช้ทั้งหมด",
            "เวลาที่ต้องใช้พัสดุ",
            "อ้างถึงบันทึกขออนุมัติหลักการเลขที่ (ถ้ามี)"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากบันทึกข้อความ 'ขออนุมัติจัดหาพัสดุ' ระบุ: "
            "1. เลขที่เอกสาร 2. วันที่ทำเอกสาร 3. เรื่อง 4. เหตุผลที่ต้องจัดหา 5. รายละเอียดของพัสดุ "
            "6. วงเงินที่ใช้ทั้งหมด 7. เวลาที่ต้องใช้พัสดุ 8. เลขที่หลักการเดิมที่อ้างอิงถึง หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "procurement_attachment": {
        "title": "เอกสารประกอบการขออนุมัติจัดหา",
        "group": "official_reimbursement",
        "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
        "keys": [
            "แนบท้ายบันทึกเอกสารเลขอะไร",
            "รายละเอียดพัสดุหรือรายการเปรียบเทียบราคา",
            "ยอดรวม"
        ],
        "system_prompt": (
            "สกัดข้อมูลจากเอกสารประกอบการขออนุมัติจัดหา ระบุ: "
            "1. แนบท้ายบันทึกเอกสารเลขอะไร 2. รายละเอียดพัสดุหรือเปรียบเทียบราคา 3. ยอดรวม "
            "หากไม่มีระบุให้ตอบว่า 'ไม่มีระบุ'"
        )
    },
    "general_receipt": {
        "title": "ใบเสร็จรับเงิน / ใบกำกับภาษีทั่วไป (ร้านค้า/บริษัท)",
        "group": "supplementary",
        "group_title": "เอกสารประกอบภายนอก (หมวดเสริม)",
        "keys": [
            "ชื่อร้านค้าหรือบริษัทผู้ขาย",
            "เลขประจำตัวผู้เสียภาษี 13 หลัก",
            "วันที่ออกเอกสาร",
            "ยอดรวมก่อนภาษี (Subtotal)",
            "ภาษีมูลค่าเพิ่ม 7% (VAT)",
            "ยอดเงินรวมทั้งสิ้น (Total Amount)",
            "รายการสินค้าและบริการ",
            "การตรวจสอบความถูกต้องของยอดเงิน (Cross-check: รวมก่อนภาษี + VAT 7% เท่ากับยอดเงินรวมทั้งสิ้นหรือไม่)"
        ],
        "system_prompt": (
            "คุณเป็นผู้เชี่ยวชาญการตรวจเอกสารการเงิน สกัดข้อมูลจากใบเสร็จรับเงินหรือใบกำกับภาษีอย่างแม่นยำ "
            "ระบุชื่อร้านค้า เลขประจำตัวผู้เสียภาษี 13 หลัก วันที่ ยอดรวมก่อนภาษี VAT 7% และยอดสุทธิรวมทั้งสิ้น "
            "พร้อมทั้งตรวจสอบความถูกต้องทางคณิตศาสตร์ (Cross-check: ยอดรวมก่อนภาษี + VAT 7% เท่ากับยอดรวมทั้งสิ้นหรือไม่ และแสดงวิธีคำนวณพร้อมผลต่าง) "
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


def configure_box_vertical_expansion(top_ratio: float = 0.0, bottom_ratio: float = 0.0):
    """
    Dynamically configures vertical bounding box expansion for PaddleX OCR text lines:
    - top_ratio: percentage of line height to expand upwards (e.g. 0.70 for 70%)
    - bottom_ratio: percentage of line height to expand downwards (e.g. 0.65 for 65%)
    When ratios are 0.0, uses default tight cropping.
    """
    try:
        import cv2
        import numpy as np
        import paddlex.inference.pipelines.components.common.crop_image_regions as cir

        if top_ratio <= 0.0 and bottom_ratio <= 0.0:
            if hasattr(cir.CropByPolys, "_unpatched_get_minarea_rect_crop"):
                cir.CropByPolys.get_minarea_rect_crop = cir.CropByPolys._unpatched_get_minarea_rect_crop
            return

        if not hasattr(cir.CropByPolys, "_unpatched_get_minarea_rect_crop"):
            cir.CropByPolys._unpatched_get_minarea_rect_crop = cir.CropByPolys.get_minarea_rect_crop

        def expanded_get_minarea_rect_crop(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
            bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
            pts = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

            index_a, index_b, index_c, index_d = 0, 1, 2, 3
            if pts[1][1] > pts[0][1]:
                index_a = 0
                index_d = 1
            else:
                index_a = 1
                index_d = 0
            if pts[3][1] > pts[2][1]:
                index_b = 2
                index_c = 3
            else:
                index_b = 3
                index_c = 2

            box = np.array([pts[index_a], pts[index_b], pts[index_c], pts[index_d]], dtype=np.float32)
            v_left = box[0] - box[3]
            v_right = box[1] - box[2]

            box_expanded = box.copy()
            box_expanded[0] = box[0] + top_ratio * v_left
            box_expanded[1] = box[1] + top_ratio * v_right
            box_expanded[3] = box[3] - bottom_ratio * v_left
            box_expanded[2] = box[2] - bottom_ratio * v_right

            H_img, W_img = img.shape[:2]
            box_expanded[:, 0] = np.clip(box_expanded[:, 0], 0, W_img - 1)
            box_expanded[:, 1] = np.clip(box_expanded[:, 1], 0, H_img - 1)

            crop_img = self.get_rotate_crop_image(img, box_expanded)
            return crop_img

        cir.CropByPolys.get_minarea_rect_crop = expanded_get_minarea_rect_crop
    except Exception as e:
        import logging
        logging.getLogger("pp_chatocr").warning(f"Could not hook CropByPolys for vertical expansion: {e}")


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
        rec_batch_size: int = 1,
        expand_box_top_ratio: float = 0.0,
        expand_box_bottom_ratio: float = 0.0,
        device: str = "cpu"
    ):
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.api_key = api_key
        self.rec_batch_size = rec_batch_size
        self.expand_box_top_ratio = float(os.getenv("OCR_EXPAND_TOP_RATIO", str(expand_box_top_ratio)))
        self.expand_box_bottom_ratio = float(os.getenv("OCR_EXPAND_BOTTOM_RATIO", str(expand_box_bottom_ratio)))
        self.device = device
        self._pipeline = None
        self._visual_cache: Dict[str, Dict[str, Any]] = {}
        self._validator = FinancialDocumentValidator()

        # Apply vertical bounding box expansion if requested
        if self.expand_box_top_ratio > 0.0 or self.expand_box_bottom_ratio > 0.0:
            configure_box_vertical_expansion(self.expand_box_top_ratio, self.expand_box_bottom_ratio)

    @staticmethod
    def get_supported_templates() -> List[Dict[str, Any]]:
        """Returns metadata of all supported document type templates categorized into official and supplementary."""
        results = []
        for key, conf in PP_CHATOCR_TEMPLATES.items():
            results.append({
                "id": key,
                "title": conf["title"],
                "group": conf.get("group", "official_reimbursement"),
                "group_title": conf.get("group_title", "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)"),
                "keys": conf["keys"],
                "system_prompt": conf["system_prompt"]
            })
        return results

    def get_pipeline(self):
        """Lazy initialization of PP-ChatOCRv4 pipeline with Thai OCR optimizations."""
        if self._pipeline is None:
            cfg = load_pipeline_config('PP-ChatOCRv4-doc')
            cfg['use_mllm_predict'] = False
            
            # Disable unwarping and doc orientation classifier to avoid distortion on flat documents
            cfg['SubPipelines']['LayoutParser']['use_doc_preprocessor'] = False

            # Use PicoDet-S_layout_3cls for maximum stability on CPU
            cfg['SubPipelines']['LayoutParser']['SubModules']['LayoutDetection']['model_name'] = 'PicoDet-S_layout_3cls'
            
            # CRITICAL FOR THAI OCR: Disable textline orientation classifier!
            # Chinese/English orientation models misclassify Thai vowels/marks as upside-down and rotate text 180 deg,
            # causing numbers like "อว 78.101/334" to turn into garbled "DEE/IOT'8L CO"
            cfg['SubPipelines']['LayoutParser']['SubPipelines']['GeneralOCR']['use_textline_orientation'] = False

            # Upgrade detector to PP-OCRv6_medium_det with optimal CPU resolution limit (1600px)
            cfg['SubPipelines']['LayoutParser']['SubPipelines']['GeneralOCR']['SubModules']['TextDetection']['model_name'] = 'PP-OCRv6_medium_det'
            cfg['SubPipelines']['LayoutParser']['SubPipelines']['GeneralOCR']['SubModules']['TextDetection']['limit_side_len'] = 1600

            # Thai OCR recognition model and batch size
            # (Empirical benchmark: batch_size=1 is ~2x faster on CPU due to zero-padding elimination;
            # on GPU, batch_size=8 or 16 leverages CUDA tensor cores)
            cfg['SubPipelines']['LayoutParser']['SubPipelines']['GeneralOCR']['SubModules']['TextRecognition']['model_name'] = 'th_PP-OCRv5_mobile_rec'
            cfg['SubPipelines']['LayoutParser']['SubPipelines']['GeneralOCR']['SubModules']['TextRecognition']['batch_size'] = self.rec_batch_size
            
            # Local Ollama LLM endpoint
            cfg['SubModules']['LLM_Chat']['base_url'] = self.ollama_url
            cfg['SubModules']['LLM_Chat']['model_name'] = self.llm_model
            cfg['SubModules']['LLM_Chat']['api_key'] = self.api_key

            pp_opt = PaddlePredictorOption()
            pp_opt.enable_new_ir = False

            self._pipeline = create_pipeline(config=cfg, device=self.device, pp_option=pp_opt)
        return self._pipeline

    def visual_predict(self, image_path: Union[str, Path], use_cache: bool = True, max_side_limit: int = 1600) -> Dict[str, Any]:
        """
        Run layout analysis and OCR perception on the document image.
        Caches results by image path to enable instant multi-turn chatting.
        Automatically scales images down to max_side_limit (default 1600px) to maximize CPU speed.
        """
        from PIL import Image
        img_p = Path(image_path).resolve()
        img_str = str(img_p)
        if use_cache and img_str in self._visual_cache:
            return self._visual_cache[img_str]

        # Automatic CPU-friendly resolution normalization
        target_eval_path = img_str
        temp_scaled_path = None
        try:
            with Image.open(img_p) as pil_im:
                w, h = pil_im.size
                max_side = max(w, h)
                if max_side > max_side_limit:
                    scale = max_side_limit / float(max_side)
                    new_w = int(round(w * scale))
                    new_h = int(round(h * scale))
                    scaled_im = pil_im.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    cache_dir = img_p.parent / ".scaled_cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    temp_scaled_path = cache_dir / f"scaled_{max_side_limit}_{img_p.name}"
                    scaled_im.save(temp_scaled_path)
                    target_eval_path = str(temp_scaled_path)
        except Exception:
            target_eval_path = img_str

        pipeline = self.get_pipeline()
        raw_res_list = list(pipeline.visual_predict(target_eval_path))
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

        # Step 3: Extract OCR text lines for context & inspection
        ocr_text_lines = []
        normal_texts = sub_vi.get("normal_text_dict", {})
        if isinstance(normal_texts, dict):
            for block in normal_texts.values():
                if isinstance(block, list):
                    for item in block:
                        for line in str(item).split("\n"):
                            l_strip = line.strip()
                            if l_strip:
                                ocr_text_lines.append(l_strip)
                elif isinstance(block, str):
                    for line in block.split("\n"):
                        l_strip = line.strip()
                        if l_strip:
                            ocr_text_lines.append(l_strip)

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
            "document_title": img_path.name,
            "document_type_name": template["title"],
            "questions": effective_questions,
            "system_prompt_used": effective_prompt,
            "chat_answers": chat_answers,
            "knowledge_payload": knowledge_payload,
            "ocr_text_snippet": " ".join(ocr_text_lines[:20]),
            "ocr_full_text": "\n".join(ocr_text_lines),
            "ocr_text_lines": ocr_text_lines,
            "ocr_line_count": len(ocr_text_lines),
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
        principle_doc_no = None
        vendor_or_requester = None
        vendor_tax_id = None
        doc_date_str = None
        total_amount = None
        subtotal = None
        vat_amount = None
        cross_check_text = None
        currency = "THB"

        for key, val in chat_answers.items():
            if not isinstance(val, str) or val in ("ไม่มีระบุ", "未知", "null", ""):
                continue
            
            k_lower = key.lower()

            # Cross-check answer: record and skip further numeric field matching
            if any(term in k_lower for term in ["cross-check", "ตรวจสอบความถูกต้อง", "ตรวจสอบยอดเงิน", "ตรวจสอบ"]):
                if not cross_check_text:
                    cross_check_text = val.strip()
                continue

            # Principle Document Reference (เลขอ้างอิงหลักการ / เอกสารอ้างอิงเดิม)
            if any(term in k_lower for term in ["หลักการ", "ที่มาของรายการ", "อ้างถึงบันทึก", "ตามหนังสือขออนุมัติหลักการ", "ตามหนังสืออนุมัติหลักการ", "ตามหนังสืออนุมัติเบิกจ่าย", "แนบท้ายบันทึกเอกสาร"]):
                if not principle_doc_no:
                    memo_m = re.search(r"(?:อว|ที่\s*อว)?\s*[\d\.\w\/-]+", val)
                    if memo_m and len(memo_m.group(0).strip()) > 3:
                        principle_doc_no = memo_m.group(0).strip()
                    else:
                        principle_doc_no = val.strip()

            # 1. Document No (เลขที่เอกสาร)
            if any(term in k_lower for term in ["เลขที่", "doc_no", "no."]) and not any(term in k_lower for term in ["หลักการ", "อ้างถึง", "ผู้เสียภาษี", "ตามหนังสือ", "แนบท้าย"]):
                if not doc_no:
                    doc_no = val.strip()

            # 2. Requester, Approver, Vendor or Officer (ผู้เบิก / ใครเป็นคนเบิก / ผู้จ่ายเงิน / ผู้ตรวจรับ)
            if any(term in k_lower for term in ["ผู้ทำการเบิก", "ใครเป็นคนเบิก", "ผู้ยืม", "ผู้ขอ", "ร้านค้า", "บริษัท", "ผู้ขาย", "ผู้จ่ายเงิน", "ผู้ตรวจรับพัสดุ", "requester", "vendor"]):
                if not vendor_or_requester:
                    vendor_or_requester = val.strip()

            # 3. Tax ID
            if any(term in k_lower for term in ["เลขประจำตัวผู้เสียภาษี", "tax id", "tax_id"]):
                tax_match = re.search(r"\d{13}", val.replace("-", "").replace(" ", ""))
                vendor_tax_id = tax_match.group(0) if tax_match else val.strip()

            # 4. Date (วันที่ทำเอกสาร / วันที่ส่งเอกสาร / วัน/เดือน/ปี)
            if any(term in k_lower for term in ["วันที่", "date", "วันเดือนปี", "วัน/เดือน/ปี", "วันที่ทำเอกสาร", "วันที่ส่งเอกสาร"]):
                if not doc_date_str:
                    doc_date_str = val.strip()

            # Subtotal (ก่อนภาษี)
            if any(term in k_lower for term in ["ก่อนภาษี", "subtotal", "รวมก่อนภาษี"]):
                amt_match = re.search(r"[\d,]+(?:\.\d{1,2})?", val)
                if amt_match:
                    try:
                        subtotal = float(amt_match.group(0).replace(",", ""))
                    except ValueError:
                        pass

            # VAT (ภาษีมูลค่าเพิ่ม)
            if any(term in k_lower for term in ["ภาษีมูลค่าเพิ่ม", "vat"]):
                amt_match = re.search(r"[\d,]+(?:\.\d{1,2})?", val)
                if amt_match:
                    try:
                        vat_amount = float(amt_match.group(0).replace(",", ""))
                    except ValueError:
                        pass

            # 5. Total Amount (ยอดรวม / ยอดรวมที่อนุมัติ / วงเงินที่ใช้ทั้งหมด)
            if any(term in k_lower for term in ["ยอดรวม", "ยอดรวมทั้งสิ้น", "ยอดเงินทั้งสิ้น", "ยอดรวมเงิน", "ยอดเงินรวม", "รวมทั้งสิ้น", "total amount", "total", "สุทธิ", "ขอยืม", "ขอเบิก", "วงเงินงบประมาณ", "งบประมาณที่ขออนุมัติ", "วงเงินที่ใช้ทั้งหมด", "ยอดรวมที่อนุมัติ"]) and not any(term in k_lower for term in ["ก่อนภาษี", "subtotal"]):
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

        # Mathematical Reconciliation Evaluation (Hybrid: Algorithmic + Formula Scanner + LLM Reasoning)
        reconcile_status = "unverified"
        reconcile_details = "ไม่มีข้อมูลตัวเลขเพียงพอสำหรับการตรวจสอบยอดเงิน"
        is_balanced = False
        calculated_total = None
        diff = 0.0

        # Algorithmic arithmetic expression scanner for rates and quantities (e.g. 600 x 2, 600 บาท จำนวน 2 คน)
        all_text_corpus = " ".join([str(v) for v in chat_answers.values()] + ocr_text_lines[:40])
        formula_matches = []

        # Pattern 1: 600 x 2 or 600 * 2 or 600.00 x 2 or 20 × 135
        for m in re.finditer(r"([\d,]+(?:\.\d{1,2})?)\s*(?:บาท|.-)?\s*(?:x|\*|คูณ|\@|\u00d7)\s*([\d,]+(?:\.\d{1,2})?)", all_text_corpus, re.IGNORECASE):
            try:
                n1 = float(m.group(1).replace(",", ""))
                n2 = float(m.group(2).replace(",", ""))
                if n1 > 0 and n2 > 0 and (n1 >= 1 and n2 >= 1):
                    formula_matches.append((n1, n2, round(n1 * n2, 2)))
            except ValueError:
                pass

        # Pattern 2: 600 บาท จำนวน 2 คน / 600 บาท (2 คน) / ละ 600 บาท จำนวน 2 คน / 600 บาท/วัน จำนวน 2 วัน
        for m in re.finditer(r"([\d,]+(?:\.\d{1,2})?)\s*(?:บาท|.-)?\s*(?:ต่อ|/|ละ)?\s*(?:คน|วัน|เที่ยว|รายการ|ชิ้น|ห้อง|มื้อ|ชุด|ราย)?\s*(?:จำนวน|\()\s*([\d,]+)\s*(?:คน|วัน|เที่ยว|รายการ|ชิ้น|ห้อง|มื้อ|ชุด|ราย|\))?", all_text_corpus):
            try:
                n1 = float(m.group(1).replace(",", ""))
                n2 = float(m.group(2).replace(",", ""))
                if n1 > 1 and n2 > 0:
                    formula_matches.append((n1, n2, round(n1 * n2, 2)))
            except ValueError:
                pass

        # Pattern 3: จำนวน 2 คน คนละ 600 บาท / จำนวน 2 วัน ละ 600 บาท
        for m in re.finditer(r"(?:จำนวน)?\s*([\d,]+)\s*(?:คน|วัน|เที่ยว|รายการ|ชิ้น|ห้อง|มื้อ|ชุด|ราย)\s*(?:ๆ\s*ละ|คนละ|ละ|อัตราละ|ต่อคนละ)\s*([\d,]+(?:\.\d{1,2})?)", all_text_corpus):
            try:
                n2 = float(m.group(1).replace(",", ""))
                n1 = float(m.group(2).replace(",", ""))
                if n1 > 0 and n2 > 0:
                    formula_matches.append((n1, n2, round(n1 * n2, 2)))
            except ValueError:
                pass

        if subtotal is not None and vat_amount is not None and total_amount is not None:
            calculated_total = round(subtotal + vat_amount, 2)
            diff = round(abs(calculated_total - total_amount), 2)
            if diff <= 0.05:
                reconcile_status = "passed"
                is_balanced = True
                reconcile_details = f"ยอดเงินตรงกันสมบูรณ์: รวมก่อนภาษี ({subtotal:,.2f}) + VAT ({vat_amount:,.2f}) = {total_amount:,.2f} บาท"
            else:
                reconcile_status = "discrepancy"
                is_balanced = False
                reconcile_details = f"ตรวจพบผลต่าง: รวมก่อนภาษี + VAT ได้ {calculated_total:,.2f} บาท แต่ยอดรวมระบุ {total_amount:,.2f} บาท (ต่างกัน {diff:,.2f} บาท)"
        elif formula_matches and total_amount is not None:
            # Check the best matching formula against total amount
            matched_exact = False
            first_f = formula_matches[0]
            for n1, n2, f_total in formula_matches:
                if abs(f_total - total_amount) <= 0.05:
                    reconcile_status = "passed"
                    is_balanced = True
                    calculated_total = f_total
                    diff = 0.0
                    reconcile_details = f"ยอดเงินตรงกันสมบูรณ์: คำนวณสูตร {n1:,.2f} x {int(n2) if n2.is_integer() else n2} = {total_amount:,.2f} บาท"
                    matched_exact = True
                    break
            
            if not matched_exact:
                n1, n2, f_total = first_f
                calculated_total = f_total
                diff = round(abs(f_total - total_amount), 2)
                reconcile_status = "discrepancy"
                is_balanced = False
                reconcile_details = f"ตรวจพบยอดเงินไม่ตรงกัน (Discrepancy): คำนวณสูตร {n1:,.2f} x {int(n2) if n2.is_integer() else n2} = {f_total:,.2f} บาท แต่ระบุยอดรวม {total_amount:,.2f} บาท (ต่างกัน {diff:,.2f} บาท)"
        elif subtotal is not None and total_amount is not None and vat_amount is None:
            diff = round(abs(subtotal - total_amount), 2)
            if diff <= 0.05:
                reconcile_status = "passed"
                is_balanced = True
                calculated_total = subtotal
                reconcile_details = f"ยอดรวมก่อนภาษีตรงกับยอดสุทธิ: {total_amount:,.2f} บาท (ไม่มี VAT)"
            else:
                calculated_total = subtotal
                reconcile_status = "unverified"
                reconcile_details = f"ยอดก่อนภาษี {subtotal:,.2f} บาท vs ยอดสุทธิ {total_amount:,.2f} บาท"
        elif cross_check_text:
            unverified_keywords = ["ไม่มีรายการย่อย", "ไม่พบรายการย่อย", "ไม่สามารถดำเนินการ", "ไม่สามารถตรวจสอบ", "ไม่มีระบุ", "ไม่ปรากฏรายการ"]
            discrepancy_keywords = ["ไม่ตรง", "ต่างกัน", "คลาดเคลื่อน", "ไม่สอดคล้อง", "เกิน", "ขาด", "discrepancy", "unmatched", "mismatch", "1,200", "1200"]
            pass_keywords = ["คำนวณตรงกัน", "รวมกันได้ตรงกับ", "ยอดตรงกัน", "ตรงกับยอด", "เท่ากับยอด", "ได้ตรงกับ", "สอดคล้องกัน", "ตรงตามที่ระบุ", "ยอดรวมถูกต้อง", "matches", "balanced", "reconciled"]

            has_unverified = any(kw in cross_check_text for kw in unverified_keywords)
            has_discrepancy = any(kw in cross_check_text for kw in discrepancy_keywords)
            has_passed = any(kw in cross_check_text for kw in pass_keywords)

            if has_discrepancy:
                reconcile_status = "discrepancy"
                is_balanced = False
                reconcile_details = f"ตรวจพบความคลาดเคลื่อน: {cross_check_text}"
            elif has_unverified:
                reconcile_status = "unverified"
                is_balanced = False
                if total_amount is not None:
                    reconcile_details = f"มียอดเงินระบุ {total_amount:,.2f} บาท (ยังไม่สามารถกระทบยอดได้เนื่องจากไม่พบรายการย่อย)"
                else:
                    reconcile_details = cross_check_text
            elif has_passed and not has_discrepancy:
                reconcile_status = "passed"
                is_balanced = True
                reconcile_details = f"ตรวจสอบผ่านการคำนวณของ AI: {cross_check_text}"
            else:
                reconcile_status = "verified_by_llm"
                is_balanced = False
                reconcile_details = cross_check_text
        elif total_amount is not None:
            reconcile_status = "unverified"
            is_balanced = False
            reconcile_details = f"มียอดเงินระบุ {total_amount:,.2f} บาท (ไม่พบรายการย่อยสำหรับกระทบยอด)"

        # Status badge label in Thai for embedding (clean without emojis)
        status_thai_map = {
            "passed": "ตรวจสอบถูกต้อง (Reconciled)",
            "discrepancy": "พบยอดเงินไม่ตรงกัน (Discrepancy)",
            "verified_by_llm": "ยืนยันผ่าน AI Cross-Check",
            "unverified": "ยังไม่ได้ตรวจสอบ (Unverified)"
        }
        status_thai = status_thai_map.get(reconcile_status, reconcile_status)

        # Principle doc no resolution
        if document_type == "principle_approval_request":
            principle_doc_no = doc_no or principle_doc_no
        elif not principle_doc_no:
            memo_match = re.search(r"(?:อว|ที่\s*อว)\s*[\d\.\w\/-]+", all_text_corpus)
            if memo_match and len(memo_match.group(0).strip()) > 3:
                principle_doc_no = memo_match.group(0).strip()

        # Build Clean Embed Text
        md_lines = [
            f"# เอกสารการเงิน: {source_filename}",
            f"- **ชื่อเอกสาร (Document Title):** {source_filename}",
            f"- **ประเภทเอกสาร:** {doc_title}",
            f"- **เลขที่เอกสาร:** {doc_no or 'ไม่ระบุ'}",
        ]
        if principle_doc_no:
            md_lines.append(f"- **อ้างอิงเอกสารหลักการเลขที่:** {principle_doc_no}")
        md_lines.extend([
            f"- **บุคคล/หน่วยงาน/ร้านค้า:** {vendor_or_requester or 'ไม่ระบุ'}",
            f"- **วันที่เอกสาร:** {doc_date_str or 'ไม่ระบุ'}" + (f" (ISO: {doc_date_iso})" if doc_date_iso else ""),
            f"- **ยอดเงินรวม:** {f'{total_amount:,.2f} บาท' if total_amount is not None else 'ไม่ระบุ'}",
            f"- **การตรวจสอบความถูกต้องของยอดเงิน (Reconciliation):** {status_thai} - {reconcile_details}",
        ])
        if subtotal is not None:
            md_lines.append(f"- **ยอดรวมก่อนภาษี (Subtotal):** {subtotal:,.2f} บาท")
        if vat_amount is not None:
            md_lines.append(f"- **ภาษีมูลค่าเพิ่ม (VAT):** {vat_amount:,.2f} บาท")
        if vendor_tax_id:
            md_lines.append(f"- **เลขประจำตัวผู้เสียภาษี:** {vendor_tax_id}")

        md_lines.append("")
        md_lines.append("## ข้อมูลที่สกัดได้ตาม Template (PP-ChatOCRv4):")
        for q, a in chat_answers.items():
            md_lines.append(f"- **{q}:** {a}")

        embed_text = "\n".join(md_lines)

        filter_metadata = {
            "principle_doc_no": principle_doc_no,
            "document_type": document_type,
            "document_title": source_filename,
            "document_type_name": doc_title,
            "doc_no": doc_no,
            "vendor_or_requester": vendor_or_requester,
            "vendor_tax_id": vendor_tax_id,
            "doc_date_iso": doc_date_iso,
            "subtotal": subtotal,
            "vat": vat_amount,
            "total_amount": total_amount,
            "currency": currency,
            "source_filename": source_filename,
            "math_reconciliation": {
                "status": reconcile_status,
                "is_balanced": is_balanced,
                "details": reconcile_details,
                "subtotal": subtotal,
                "vat": vat_amount,
                "stated_total": total_amount,
                "calculated_total": calculated_total,
                "difference": diff
            },
            "extracted_at": datetime.now().isoformat(),
            "extracted_by": f"PP-ChatOCRv4 ({self.llm_model})"
        }

        return {
            "embed_text": embed_text,
            "filter_metadata": filter_metadata
        }
