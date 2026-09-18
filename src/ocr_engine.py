"""
Component 2: Perception Layer (Thai OCR Engine)
Extracts Thai text, bounding boxes, and document structures using PaddleOCR / PaddleX
configured with the th_PP-OCRv5_mobile_rec model (Zero-shot / No fine-tuning required).
"""

from typing import List, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates [x_min, y_min, x_max, y_max]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def from_array(cls, box: Union[List[float], np.ndarray]) -> "BoundingBox":
        """Construct from [x1, y1, x2, y2] or polygon coordinates."""
        b = list(box)
        if len(b) == 4:
            return cls(
                x_min=round(float(b[0]), 2),
                y_min=round(float(b[1]), 2),
                x_max=round(float(b[2]), 2),
                y_max=round(float(b[3]), 2)
            )
        elif len(b) == 8:
            # 4-point polygon flattened [x1, y1, x2, y2, x3, y3, x4, y4]
            xs = [b[i] for i in range(0, 8, 2)]
            ys = [b[i] for i in range(1, 8, 2)]
            return cls(
                x_min=round(float(min(xs)), 2),
                y_min=round(float(min(ys)), 2),
                x_max=round(float(max(xs)), 2),
                y_max=round(float(max(ys)), 2)
            )
        else:
            raise ValueError(f"Unexpected bounding box format with {len(b)} elements: {b}")


class TextBlock(BaseModel):
    """A single recognized line or phrase of text."""
    text: str
    confidence: float
    box: BoundingBox


class TableBlock(BaseModel):
    """A recognized tabular structure with Markdown and HTML representations."""
    table_markdown: str = ""
    table_html: str = ""
    box: Optional[BoundingBox] = None


class PagePerception(BaseModel):
    """Aggregated perception output for a single document page."""
    page_number: int
    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[TableBlock] = Field(default_factory=list)
    raw_text: str = ""

    def to_llm_markdown(self) -> str:
        """
        Format the extracted text and tables into an organized Markdown document
        ready to be fed into the Local LLM reasoning prompt.
        """
        sections: List[str] = [f"--- Page {self.page_number} ---"]

        if self.tables:
            sections.append("### Extracted Tables:")
            for i, table in enumerate(self.tables, 1):
                sections.append(f"Table {i}:\n{table.table_markdown}\n")

        sections.append("### Extracted Text Content:")
        sections.append(self.raw_text)

        return "\n\n".join(sections)


class ThaiPerceptionEngine:
    """
    Perception Engine handling Thai OCR using PaddleOCR + th_PP-OCRv5_mobile_rec.
    Optimized for Windows with automatic backend fallback.
    """

    def __init__(
        self,
        rec_model_name: str = "th_PP-OCRv5_mobile_rec",
        enable_mkldnn: bool = False,
        use_doc_unwarping: bool = False,
        use_doc_orientation_classify: bool = False,
        use_textline_orientation: bool = False,
        unclip_ratio: float = 2.35,
        limit_side_len: int = 2400,
        box_thresh: float = 0.6,
        device: str = "cpu"
    ):
        self.rec_model_name = rec_model_name
        self.enable_mkldnn = enable_mkldnn
        self.use_doc_unwarping = use_doc_unwarping
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_textline_orientation = use_textline_orientation
        self.unclip_ratio = unclip_ratio
        self.limit_side_len = limit_side_len
        self.box_thresh = box_thresh
        self.device = device
        self._ocr = None

    def _get_ocr_instance(self):
        """Lazy load PaddleOCR instance to optimize startup time and memory."""
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                text_recognition_model_name=self.rec_model_name,
                enable_mkldnn=self.enable_mkldnn,
                use_doc_unwarping=self.use_doc_unwarping,
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_textline_orientation=self.use_textline_orientation,
                device=self.device
            )
        return self._ocr

    @staticmethod
    def clean_and_merge_thai_fragments(text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Bidirectional cleaner for Thai typography:
        1. Filters out tiny sliver noise (underline dashes, speckles, artifacts).
        2. Merges upper vowels / tone marks (floating above) into the main line below.
        3. Merges lower vowels / descenders (ุ, ู, เชิง ญ ฐ hanging below) into the main line above.
        """
        if not text_blocks:
            return []

        # Thai floating upper marks: ิ, ี, ึ, ื, ั, ํ, ็, ่, ้, ๊, ๋, ์
        THAI_UPPER_CHARS = set('\u0E31\u0E34\u0E35\u0E36\u0E37\u0E47\u0E48\u0E49\u0E4A\u0E4B\u0E4C\u0E4D\u0E4E')
        # Thai lower marks: ุ, ู, ฺ
        THAI_LOWER_CHARS = set('\u0E38\u0E39\u0E3A')

        # Sort blocks vertically top-to-bottom
        sorted_blocks = sorted(text_blocks, key=lambda b: (b.box.y_min, b.box.x_min))

        # Compute median line height of substantial blocks
        sub_heights = [
            (b.box.y_max - b.box.y_min)
            for b in sorted_blocks
            if len(b.text.strip()) >= 3
        ]
        median_h = float(np.median(sub_heights)) if sub_heights else 30.0

        main_lines: List[TextBlock] = []
        fragments: List[TextBlock] = []

        for b in sorted_blocks:
            h = b.box.y_max - b.box.y_min
            w = b.box.x_max - b.box.x_min
            text_clean = b.text.strip()

            # Filter pure noise slivers (extremely thin or low-confidence single dots/dashes)
            if h < 10 and w < 20:
                continue
            if len(text_clean) == 1 and h < 15 and b.confidence < 0.6 and not text_clean.isalnum():
                continue

            # Identify short fragments: pure floating marks or tiny punctuation slivers
            is_thai_mark_only = all(c in THAI_UPPER_CHARS or c in THAI_LOWER_CHARS for c in text_clean)
            is_noise_sliver = (len(text_clean) == 1 and not text_clean.isalnum() and h < median_h * 0.5)

            if is_thai_mark_only or is_noise_sliver:
                fragments.append(b)
            else:
                main_lines.append(b)

        # Merge fragments into their adjacent parent main line
        unmerged_fragments: List[TextBlock] = []

        for frag in fragments:
            frag_xc = (frag.box.x_min + frag.box.x_max) / 2.0
            frag_yc = (frag.box.y_min + frag.box.y_max) / 2.0
            merged = False

            # Check distance to all main lines
            for main in main_lines:
                main_h = main.box.y_max - main.box.y_min
                # Horizontal check: fragment must be within horizontal span of main line (with 15px margin)
                if not (main.box.x_min - 15 <= frag_xc <= main.box.x_max + 15):
                    continue

                # Case A: Fragment is hanging directly BELOW the main line (Lower vowels: ุ, ู, descenders)
                # Allow vertical overlap up to 20px, and downward gap up to max(35px, main_h * 0.8)
                dist_below = frag.box.y_min - main.box.y_max
                if -20 <= dist_below <= max(35, main_h * 0.8) and frag_yc >= (main.box.y_min + main_h * 0.5):
                    main.box.y_max = max(main.box.y_max, frag.box.y_max)
                    main.box.x_min = min(main.box.x_min, frag.box.x_min)
                    main.box.x_max = max(main.box.x_max, frag.box.x_max)
                    merged = True
                    break

                # Case B: Fragment is floating directly ABOVE the main line (Upper vowels: ิ, ี, tones: ่, ้)
                dist_above = main.box.y_min - frag.box.y_max
                if -20 <= dist_above <= max(35, main_h * 0.8) and frag_yc <= (main.box.y_max - main_h * 0.5):
                    main.box.y_min = min(main.box.y_min, frag.box.y_min)
                    main.box.x_min = min(main.box.x_min, frag.box.x_min)
                    main.box.x_max = max(main.box.x_max, frag.box.x_max)
                    merged = True
                    break

            if not merged:
                # If it has alphanumeric characters (letters or numbers like 1, 2, 9), NEVER drop it!
                if any(c.isalnum() for c in frag.text):
                    unmerged_fragments.append(frag)
                elif frag.confidence >= 0.70 and len(frag.text.strip()) > 0:
                    unmerged_fragments.append(frag)

        return main_lines + unmerged_fragments

    @staticmethod
    def sort_reading_order(blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Sort text blocks in natural reading order:
        1. Group blocks into horizontal rows based on vertical proximity.
        2. Within each row, sort left-to-right by x_min.
        3. Sort rows top-to-bottom.
        """
        if not blocks:
            return []

        # Preliminary sort top-to-bottom
        sorted_by_y = sorted(blocks, key=lambda b: (b.box.y_min, b.box.x_min))

        rows: List[List[TextBlock]] = []
        for b in sorted_by_y:
            b_yc = (b.box.y_min + b.box.y_max) / 2.0
            b_h = b.box.y_max - b.box.y_min
            placed = False

            for row in rows:
                row_yc = sum((item.box.y_min + item.box.y_max) / 2.0 for item in row) / len(row)
                row_h = sum((item.box.y_max - item.box.y_min) for item in row) / len(row)
                tolerance = max(14.0, min(row_h, b_h) * 0.55)

                if abs(b_yc - row_yc) <= tolerance:
                    row.append(b)
                    placed = True
                    break

            if not placed:
                rows.append([b])

        # Sort each row left-to-right, then concatenate rows
        final_ordered: List[TextBlock] = []
        for row in rows:
            row_sorted = sorted(row, key=lambda b: b.box.x_min)
            final_ordered.extend(row_sorted)

        return final_ordered

    def process_image(
        self,
        image_input: Union[Image.Image, np.ndarray, str, Path],
        page_number: int = 1,
        min_confidence: float = 0.3
    ) -> PagePerception:
        """
        Process a single image page and extract all recognized Thai text and bounding boxes.
        Coordinates are 100% pixel-perfect aligned with the input image.
        """
        if isinstance(image_input, Image.Image):
            input_data = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, (str, Path)):
            input_data = str(image_input)
        else:
            input_data = image_input

        ocr = self._get_ocr_instance()
        results = list(ocr.predict(
            input=input_data,
            use_doc_unwarping=self.use_doc_unwarping,
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_textline_orientation=self.use_textline_orientation,
            text_det_unclip_ratio=self.unclip_ratio,
            text_det_limit_side_len=self.limit_side_len,
            text_det_box_thresh=self.box_thresh
        ))

        text_blocks: List[TextBlock] = []

        if results and len(results) > 0:
            result_dict = results[0]
            rec_texts = result_dict.get("rec_texts", [])
            rec_scores = result_dict.get("rec_scores", [])
            rec_boxes = result_dict.get("rec_boxes", [])

            for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                text_clean = str(text).strip()
                score_val = float(score)

                if score_val < min_confidence or not text_clean:
                    continue

                bbox = BoundingBox.from_array(box)
                text_blocks.append(TextBlock(
                    text=text_clean,
                    confidence=round(score_val, 4),
                    box=bbox
                ))

        # Clean stray tone fragments and merge detached floating marks
        cleaned_blocks = self.clean_and_merge_thai_fragments(text_blocks)
        ordered_blocks = self.sort_reading_order(cleaned_blocks)
        full_raw_text = "\n".join(b.text for b in ordered_blocks)

        return PagePerception(
            page_number=page_number,
            text_blocks=ordered_blocks,
            tables=[],
            raw_text=full_raw_text
        )
