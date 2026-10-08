"""
Component 2: Perception Layer (Thai OCR Engine)
Extracts Thai text, bounding boxes, and document structures using PaddleOCR / PaddleX
configured with the th_PP-OCRv5_mobile_rec model (Zero-shot / No fine-tuning required).
"""

from typing import List, Optional, Union
from pathlib import Path
import os
import re
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field


def clean_dotted_lines_from_image(img_input: np.ndarray, min_chain_len: int = 3) -> np.ndarray:
    """
    Intelligent Thai official form dotted fill-in line (จุดไข่ปลา) remover:
    - Identifies small dot candidates (h <= 6, w <= 14, area <= 45).
    - Identifies text characters (h >= 9).
    - Protects genuine interior punctuation inside numbers and words
      (e.g., ',' and '.' in '2,100.00' or '333.213582.0') by ensuring dots
      sandwiched between adjacent text characters are never removed.
    - Groups remaining dots into horizontal chains where dy <= 2 and dx between 3 and 16.
    - Inpaints verified dotted lines (>= min_chain_len dots) with white.
    """
    if not isinstance(img_input, np.ndarray) or img_input.size == 0:
        return img_input

    is_gray = (len(img_input.shape) == 2)
    gray = img_input if is_gray else cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # Invert binary threshold to identify dark ink on light paper
    _, binary = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return img_input

    # 1. Classify candidate dots vs text characters
    dot_candidates = []
    text_chars = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # Dots of dotted leader lines have small height and area
        if h <= 6 and w <= 14 and area <= 45:
            dot_candidates.append({
                "id": i, "x": x, "y": y, "w": w, "h": h, "area": area,
                "cx": cx, "cy": cy, "xmin": x, "xmax": x + w, "ymin": y, "ymax": y + h
            })
        elif h >= 9:
            text_chars.append({
                "id": i, "x": x, "y": y, "w": w, "h": h, "area": area,
                "xmin": x, "xmax": x + w, "ymin": y, "ymax": y + h
            })

    if len(dot_candidates) < min_chain_len:
        return img_input

    # 2. Protect punctuation marks that sit tightly inside a word/number (e.g. '.' or ',' in '2,100.00')
    # A mark is inside a word if it is within 4px of a character to the left AND to the right
    safe_dots = []
    for d in dot_candidates:
        has_left_char = False
        has_right_char = False

        for c in text_chars:
            if not (c["ymax"] < d["ymin"] - 3 or c["ymin"] > d["ymax"] + 3):
                if 0 <= (d["xmin"] - c["xmax"]) <= 4:
                    has_left_char = True
                if 0 <= (c["xmin"] - d["xmax"]) <= 4:
                    has_right_char = True

        if has_left_char and has_right_char:
            # Protected interior punctuation
            continue

        safe_dots.append(d)

    # 3. Form horizontal chains of dots
    safe_dots.sort(key=lambda d: (round(d["cy"] / 3.0), d["cx"]))

    chains = []
    current_chain = []

    for d in safe_dots:
        if not current_chain:
            current_chain.append(d)
            continue

        prev = current_chain[-1]
        dy = abs(d["cy"] - prev["cy"])
        dx = d["cx"] - prev["cx"]

        if dy <= 2 and 3 <= dx <= 16:
            current_chain.append(d)
        else:
            if len(current_chain) >= min_chain_len:
                chains.append(current_chain)
            current_chain = [d]

    if len(current_chain) >= min_chain_len:
        chains.append(current_chain)

    if not chains:
        return img_input

    # 4. Inpaint chained dots
    erase_mask = np.zeros(gray.shape, dtype=np.uint8)
    for chain in chains:
        for d in chain:
            erase_mask[d["y"]:d["y"]+d["h"], d["x"]:d["x"]+d["w"]] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_erase = cv2.dilate(erase_mask, kernel)

    result = img_input.copy()
    if is_gray:
        result[dilated_erase == 255] = 255
    else:
        result[dilated_erase == 255] = [255, 255, 255]

    return result


def remove_lines_text_and_binarize(
    image_path: Union[str, Path, np.ndarray, Image.Image],
    thresh_bin: int = 200,
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
    hough_threshold: int = 150,
    line_thickness: int = 7,
    inpaint_radius: int = 3,
    horizontal_only: bool = True,
    return_intermediates: bool = False,
    page_num: int = 1,
) -> Union[np.ndarray, dict]:
    """
    Image preprocessing function to remove dotted fill-in lines and form baselines.

    Steps:
    1. Read image and convert to grayscale.
    2. Detect straight lines using Canny edge detection and HoughLines:
       - Canny edge detection (canny_thresh1, canny_thresh2).
       - HoughLines detection (horizontal_only: 80°-100°).
       - Draw detected lines onto line_mask with thickness 7.
    3. Inpaint lines directly:
       - Inpaint regions where line_mask == 255 with background color.
       - Binarize inpainted image with threshold thresh_bin (default 200).
    """
    import time
    t0 = time.time()

    # 1. Read image and convert to grayscale
    if isinstance(image_path, (str, Path)):
        p = Path(image_path)
        if p.suffix.lower() == ".pdf":
            from src.ingestion import DocumentIngestion
            ingestor = DocumentIngestion(target_dpi=150)
            pages = ingestor.load_document(p)
            idx = max(0, min(page_num - 1, len(pages) - 1))
            img_bgr = pages[idx].to_cv2()
        else:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                raise ValueError(f"Could not read image from path: {image_path}")
    elif isinstance(image_path, Image.Image):
        rgb_arr = np.array(image_path.convert("RGB"))
        img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    elif isinstance(image_path, np.ndarray):
        if len(image_path.shape) == 2:
            img_bgr = cv2.cvtColor(image_path, cv2.COLOR_GRAY2BGR)
        elif image_path.shape[2] == 4:
            img_bgr = cv2.cvtColor(image_path, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = image_path.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_path)}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Detect straight lines using Canny and HoughLines
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)

    line_mask = np.zeros_like(gray, dtype=np.uint8)
    line_count = 0
    line_angles = []

    if lines is not None:
        H, W = gray.shape
        diag = int(np.sqrt(H**2 + W**2)) + 500
        for line in lines:
            rho, theta = line[0]
            deg = np.degrees(theta)
            if horizontal_only and not (80.0 <= deg <= 100.0):
                continue
            line_count += 1
            if len(line_angles) < 30:
                line_angles.append(round(float(deg), 1))
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + diag * (-b))
            y1 = int(y0 + diag * a)
            x2 = int(x0 - diag * (-b))
            y2 = int(y0 - diag * a)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=line_thickness)

    # 3. Inpaint lines where line_mask == 255, then binarize
    has_lines = bool(np.any(line_mask == 255))
    if has_lines:
        inpainted_bgr = cv2.inpaint(img_bgr, line_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    else:
        inpainted_bgr = img_bgr.copy()

    inpainted_gray = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2GRAY)
    _, final_binarized = cv2.threshold(inpainted_gray, thresh_bin, 255, cv2.THRESH_BINARY)

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    if not return_intermediates:
        return final_binarized

    return {
        "final_image": final_binarized,
        "original_bgr": img_bgr,
        "gray": gray,
        "canny_edges": edges,
        "line_mask": line_mask,
        "inpainted_bgr": inpainted_bgr,
        "inpainted_gray": inpainted_gray,
        "final_binarized": final_binarized,
        "stats": {
            "elapsed_ms": elapsed_ms,
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0]),
            "detected_lines_count": line_count,
            "line_mask_pixels": int(np.sum(line_mask == 255)),
            "inpainted_pixels": int(np.sum(line_mask == 255)),
            "thresh_bin": thresh_bin,
            "hough_threshold": hough_threshold,
            "line_thickness": line_thickness,
            "horizontal_only": horizontal_only,
            "line_angles_sample": line_angles,
        }
    }


def clean_ocr_text_noise(text: str) -> str:
    """
    Post-process OCR text to clean structural artifacts:
    - Strip leading/trailing form border artifacts (dots, dashes, underscores).
    - Collapse runs of dotted fill-in lines (จุดไข่ปลา) into single spaces.
    - Collapse excessive spaces.
    Does NOT do hardcoded word replacement; semantic interpretation and spelling
    decisions are delegated to the LLM.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Strip leading/trailing dots, dashes, underscores (form lines / border artifacts)
        line_clean = re.sub(r'^[.\-_—\s]+|[.\-_—\s]+$', '', line_clean)

        # Collapse multiple internal dots (dotted fill-in lines) into space
        line_clean = re.sub(r'\.{2,}', ' ', line_clean)

        # Collapse multiple spaces
        line_clean = re.sub(r'\s{2,}', '   ', line_clean)

        if line_clean.strip():
            cleaned_lines.append(line_clean)

    return "\n".join(cleaned_lines)


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
    Perception Engine handling Thai OCR using PaddleOCR / PaddleX + th_PP-OCRv5_mobile_rec.
    Supports ONNX Runtime acceleration and automatic dotted line cleaning for Thai official forms.
    """

    def __init__(
        self,
        rec_model_name: str = "th_PP-OCRv5_mobile_rec",
        enable_mkldnn: bool = False,
        use_doc_unwarping: bool = False,
        use_doc_orientation_classify: bool = False,
        use_textline_orientation: bool = False,
        return_word_box: bool = False,
        unclip_ratio: float = 2.35,
        limit_side_len: int = 960,
        box_thresh: float = 0.6,
        rec_batch_size: int = 1,
        expand_box_top_ratio: float = 0.0,
        expand_box_bottom_ratio: float = 0.0,
        use_onnx: Optional[bool] = None,
        device: str = "cpu"
    ):
        self.rec_model_name = rec_model_name
        self.enable_mkldnn = enable_mkldnn
        self.use_doc_unwarping = use_doc_unwarping
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_textline_orientation = use_textline_orientation
        self.return_word_box = return_word_box
        self.unclip_ratio = unclip_ratio
        self.limit_side_len = limit_side_len
        self.box_thresh = box_thresh
        self.rec_batch_size = rec_batch_size
        self.expand_box_top_ratio = float(os.getenv("OCR_EXPAND_TOP_RATIO", str(expand_box_top_ratio)))
        self.expand_box_bottom_ratio = float(os.getenv("OCR_EXPAND_BOTTOM_RATIO", str(expand_box_bottom_ratio)))
        self.use_onnx = os.getenv("OCR_USE_ONNX", "true").lower() in ("1", "true", "yes") if use_onnx is None else use_onnx
        self.device = os.getenv("OCR_DEVICE", device)
        self._ocr = None

        if self.expand_box_top_ratio > 0.0 or self.expand_box_bottom_ratio > 0.0:
            try:
                from src.pp_chatocr_engine import configure_box_vertical_expansion
                configure_box_vertical_expansion(self.expand_box_top_ratio, self.expand_box_bottom_ratio)
            except Exception:
                pass

    def _get_ocr_instance(self):
        """Lazy load OCR instance (ONNX Runtime accelerated via PaddleX or PaddleOCR)."""
        if self._ocr is None:
            if self.use_onnx:
                try:
                    from paddlex.inference.pipelines import load_pipeline_config
                    from paddlex import create_pipeline
                    from paddlex.inference.models.runners.paddle_static.config.pp_option import PaddlePredictorOption

                    cfg = load_pipeline_config('OCR')
                    cfg['use_doc_preprocessor'] = False
                    cfg['use_textline_orientation'] = False
                    cfg['SubModules']['TextDetection']['model_name'] = 'PP-OCRv6_medium_det'
                    cfg['SubModules']['TextDetection']['limit_side_len'] = self.limit_side_len
                    cfg['SubModules']['TextDetection']['limit_type'] = 'max'
                    cfg['SubModules']['TextDetection']['unclip_ratio'] = self.unclip_ratio
                    cfg['SubModules']['TextDetection']['engine'] = 'onnxruntime'
                    cfg['SubModules']['TextRecognition']['model_name'] = self.rec_model_name
                    cfg['SubModules']['TextRecognition']['batch_size'] = self.rec_batch_size
                    cfg['SubModules']['TextRecognition']['engine'] = 'onnxruntime'

                    pp_opt = PaddlePredictorOption()
                    pp_opt.enable_new_ir = False

                    self._ocr = create_pipeline(config=cfg, device=self.device, pp_option=pp_opt)
                    return self._ocr
                except Exception as e:
                    print(f"[ThaiPerceptionEngine] Failed to initialize ONNX pipeline ({e}), falling back to PaddleOCR.")

            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                text_recognition_model_name=self.rec_model_name,
                text_recognition_batch_size=self.rec_batch_size,
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

    @staticmethod
    def build_row_formatted_text(blocks: List[TextBlock]) -> str:
        """
        Group blocks into natural horizontal lines based on vertical alignment.
        Items on the same line (e.g. 2-column headers or items and prices)
        are placed on the same text line separated by spaces.
        """
        if not blocks:
            return ""

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

        lines: List[str] = []
        for row in rows:
            row_sorted = sorted(row, key=lambda b: b.box.x_min)
            lines.append("   ".join(b.text for b in row_sorted))

        return "\n".join(lines)

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
            input_data = cv2.imread(str(image_input))
            if input_data is not None:
                input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            else:
                input_data = str(image_input)
        else:
            input_data = image_input

        # Automatically clean Thai official document dotted fill-in lines (จุดไข่ปลา)
        if isinstance(input_data, np.ndarray):
            input_data = clean_dotted_lines_from_image(input_data)

        ocr = self._get_ocr_instance()
        results = list(ocr.predict(
            input=input_data,
            use_doc_unwarping=self.use_doc_unwarping,
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_textline_orientation=self.use_textline_orientation,
            return_word_box=self.return_word_box,
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
                text_clean = clean_ocr_text_noise(str(text).strip())
                score_val = float(score)

                if score_val < min_confidence or not text_clean:
                    continue

                # Filter out pure noise / dotted or line fragments that have no alphanumeric content
                if re.fullmatch(r'^[.\-_—\s]+$', text_clean):
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
        full_raw_text = self.build_row_formatted_text(cleaned_blocks)
        full_raw_text = clean_ocr_text_noise(full_raw_text)

        return PagePerception(
            page_number=page_number,
            text_blocks=ordered_blocks,
            tables=[],
            raw_text=full_raw_text
        )
