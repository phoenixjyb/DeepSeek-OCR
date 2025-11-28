import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModel, AutoTokenizer
import fitz  # PyMuPDF


# Default prompt variants align with the repo examples.
PROMPTS = {
    "markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
    "free": "<image>\nFree OCR. ",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}


class OCRService:
    """Thin wrapper around the tested DeepSeek-OCR infer call."""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR") -> None:
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

        # Pin to the RTX 3090 (device 0) just like the existing scripts.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    async def load(self) -> None:
        """Load tokenizer and model once."""
        if self.model and self.tokenizer:
            return

        async with self._load_lock:
            if self.model and self.tokenizer:
                return

            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                self.model_name,
                trust_remote_code=True,
            )

            def _load_model() -> Any:
                try:
                    model = AutoModel.from_pretrained(
                        self.model_name,
                        _attn_implementation="flash_attention_2",
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                except ImportError as exc:
                    print(
                        f"[OCRService] flash_attention_2 unavailable ({exc}); falling back to default attention."
                    )
                    model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                return model.eval().cuda().to(torch.bfloat16)

            self.model = await asyncio.to_thread(_load_model)

    async def infer_image(
        self, image_path: Path, prompt_key: str, output_dir: Path
    ) -> str:
        """Run inference on a single image path."""
        await self.load()
        prompt = PROMPTS.get(prompt_key, PROMPTS["markdown"])

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / image_path.stem

        async with self._inference_lock:
            result = await asyncio.to_thread(
                self.model.infer,
                self.tokenizer,
                prompt,
                str(image_path),
                str(output_path),
                1024,  # base_size
                640,  # image_size
                True,  # crop_mode (Gundam mode like existing scripts)
                True,  # test_compress
                True,  # save_results
            )

        return str(result)


def convert_pdf_to_images(pdf_path: Path, output_folder: Path) -> List[Path]:
    """Convert PDF pages to images using the same settings as process_pdfs.py."""
    doc = fitz.open(str(pdf_path))
    output_folder.mkdir(parents=True, exist_ok=True)
    image_paths: List[Path] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = output_folder / f"page_{page_num + 1}.png"
        pix.save(str(img_path))
        image_paths.append(img_path)

    doc.close()
    return image_paths


def is_supported_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in IMAGE_EXTENSIONS or suffix in PDF_EXTENSIONS
