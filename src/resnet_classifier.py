"""Reusable inference wrapper for the latest 11-class ResNet18 checkpoint."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Union

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LATEST_RESNET_CLASSES = {
    "01Request",
    "02Approved",
    "03RequestReim",
    "04PettyCash",
    "05AdvancePayment",
    "06CertificateInsteadofReceipt",
    "07Inspectionform",
    "07_02Inspectionform",
    "08RequestMaterial",
    "09Attchedmaterial",
    "10Approved_materials",
}


class TopFractionCrop:
    def __init__(self, fraction: float):
        self.fraction = fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        height = max(1, round(image.height * self.fraction))
        return image.crop((0, 0, image.width, height))


class PadOnlyCanvas:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.width > self.width or image.height > self.height:
            raise ValueError(
                f"Cropped image {image.width}x{image.height} exceeds the model's "
                f"{self.width}x{self.height} pad-only canvas"
            )
        canvas = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        canvas.paste(
            image,
            ((self.width - image.width) // 2, (self.height - image.height) // 2),
        )
        return canvas


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
    return torch.device(requested)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ResNet checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("The ResNet checkpoint must contain a dictionary")
    return checkpoint


class ResNetDocumentClassifier:
    """Load the trained model once and return its known-class prediction."""

    def __init__(self, checkpoint_path: Union[str, Path], device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint = _load_checkpoint(self.checkpoint_path)

        if checkpoint.get("architecture") != "resnet18":
            raise ValueError(
                f"Expected a resnet18 checkpoint, found {checkpoint.get('architecture')!r}"
            )

        class_names = checkpoint.get("class_names")
        expected_classes = LATEST_RESNET_CLASSES
        if not isinstance(class_names, list) or set(class_names) != expected_classes:
            raise ValueError(
                "Checkpoint classes do not match the latest 11-class taxonomy: "
                f"found={class_names!r}, expected={sorted(expected_classes)!r}"
            )
        self.class_names = class_names

        roi_top_percent = float(checkpoint.get("roi_top_percent", 100.0))
        if not 0 < roi_top_percent <= 100:
            raise ValueError(f"Invalid ROI top percentage: {roi_top_percent}")
        if checkpoint.get("resize_mode") != "pad-only":
            raise ValueError("The latest classifier requires a pad-only checkpoint")

        pad_width = int(checkpoint.get("pad_width", 928))
        pad_height = int(checkpoint.get("pad_height", 512))
        self.transform = transforms.Compose(
            [
                TopFractionCrop(roi_top_percent / 100.0),
                PadOnlyCanvas(pad_width, pad_height),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.device = _select_device(device)
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

    @staticmethod
    def render_document_image(file_bytes: bytes) -> Image.Image:
        """Render input exactly like the classifier training/evaluation pipeline.

        OCR intentionally uses a higher-resolution 150-DPI render. The ResNet
        pad-only transform was trained from PDF pages rendered at scale 1.5, so
        feeding the OCR bitmap can exceed its fixed 928x512 top-crop canvas.
        """
        if file_bytes.startswith(b"%PDF"):
            try:
                import pypdfium2 as pdfium
            except ImportError as exc:
                raise RuntimeError(
                    "pypdfium2 is required for ResNet PDF inference"
                ) from exc
            document = pdfium.PdfDocument(file_bytes)
            if len(document) < 1:
                document.close()
                raise ValueError("The PDF contains no pages")
            page = document[0]
            try:
                return page.render(scale=1.5).to_pil().convert("RGB")
            finally:
                page.close()
                document.close()

        with Image.open(io.BytesIO(file_bytes)) as image:
            return image.convert("RGB").copy()

    def predict_document(self, file_bytes: bytes) -> dict[str, Any]:
        """Render a PDF/image with classifier preprocessing and predict it."""
        return self.predict_image(self.render_document_image(file_bytes))

    def predict_image(self, image: Image.Image) -> dict[str, Any]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().tolist()

        ranked = sorted(
            zip(self.class_names, probabilities), key=lambda pair: pair[1], reverse=True
        )
        prediction, confidence = ranked[0]
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "top_three": [
                {"class_name": class_name, "probability": float(probability)}
                for class_name, probability in ranked[:3]
            ],
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(self.class_names, probabilities)
            },
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
        }
