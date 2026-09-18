"""
Component 1: Document Ingestion and Preprocessing Module
Responsible for handling image/PDF files, rendering to high-res bitmaps,
and applying optical preprocessing (deskewing, orientation correction).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import io
import numpy as np
from PIL import Image
import cv2

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None


@dataclass
class DocumentPage:
    """Represents a single processed document page."""
    page_number: int
    image: Image.Image
    width: int
    height: int

    def to_cv2(self) -> np.ndarray:
        """Convert PIL Image to OpenCV BGR numpy array."""
        rgb_array = np.array(self.image)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


class DocumentIngestion:
    """Document loader and preprocessor for PDFs and images."""

    def __init__(self, target_dpi: int = 200):
        self.target_dpi = target_dpi
        # Standard PDF point size is 72 points per inch
        self.render_scale = target_dpi / 72.0

    def load_document(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[DocumentPage]:
        """
        Load a document from file path, bytes, or stream.
        Automatically identifies whether it is a PDF or an Image.
        """
        raw_bytes: bytes
        file_suffix = ""

        if isinstance(source, (str, Path)):
            path = Path(source)
            file_suffix = path.suffix.lower()
            with open(path, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(source, io.BytesIO):
            raw_bytes = source.getvalue()
        elif isinstance(source, bytes):
            raw_bytes = source
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # Check if PDF by extension or magic bytes (%PDF)
        is_pdf = file_suffix == ".pdf" or raw_bytes.startswith(b"%PDF")

        if is_pdf:
            return self._load_pdf(raw_bytes)
        else:
            return self._load_image(raw_bytes)

    @staticmethod
    def normalize_resolution(pil_image: Image.Image, max_dim_limit: int = 2400) -> Image.Image:
        """Scale down oversized images (> 2400px) proportionally to prevent OCR detection clipping and coordinate shift."""
        w, h = pil_image.size
        max_side = max(w, h)
        if max_side > max_dim_limit:
            scale = max_dim_limit / float(max_side)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            return pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return pil_image

    def _load_pdf(self, pdf_bytes: bytes) -> List[DocumentPage]:
        """Render each page of a PDF as a high-resolution PIL Image."""
        if pdfium is None:
            raise RuntimeError("pypdfium2 is required for PDF processing. Run: pip install pypdfium2")

        pdf = pdfium.PdfDocument(pdf_bytes)
        pages: List[DocumentPage] = []

        for page_idx in range(len(pdf)):
            page = pdf.get_page(page_idx)
            # Render page at 200 DPI for optimal OCR recognition
            bitmap = page.render(scale=self.render_scale)
            pil_image = self.normalize_resolution(bitmap.to_pil())
            page.close()

            pages.append(DocumentPage(
                page_number=page_idx + 1,
                image=pil_image,
                width=pil_image.width,
                height=pil_image.height
            ))

        pdf.close()
        return pages

    def _load_image(self, image_bytes: bytes) -> List[DocumentPage]:
        """Load single image from bytes and normalize resolution."""
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        pil_image = self.normalize_resolution(pil_image)

        return [DocumentPage(
            page_number=1,
            image=pil_image,
            width=pil_image.width,
            height=pil_image.height
        )]

    @staticmethod
    def deskew_image(pil_image: Image.Image, max_angle: float = 12.0) -> Image.Image:
        """
        Detect and correct small skew angle in scanned documents using OpenCV minAreaRect.
        Clamped to max_angle (default 12.0 deg) to prevent false aggressive rotations.
        """
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Invert colors (text becomes foreground white, background black)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find coordinates of all foreground pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return pil_image  # Not enough text pixels to reliably compute skew

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Determine true rotation angle
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # If angle is negligible or too large (e.g. false detection), skip
        if abs(angle) < 0.5 or abs(angle) > max_angle:
            return pil_image

        (h, w) = cv_img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            cv_img, rot_mat, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
