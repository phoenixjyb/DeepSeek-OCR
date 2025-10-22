# DeepSeek-OCR PDF Batch Processor

This script processes multiple PDF files from a folder and outputs OCR results.

## Features

- ✅ Batch process multiple PDFs from a folder
- ✅ Converts PDF pages to images automatically
- ✅ Runs DeepSeek-OCR on each page
- ✅ Saves individual page results and combined output
- ✅ Supports both markdown and free OCR modes
- ✅ Uses RTX 3090 GPU for fast processing

## Requirements

All requirements are already installed in the `deepseek-ocr` conda environment:
- torch==2.6.0
- transformers==4.46.3
- torchvision==0.21.0
- flash-attn==2.7.3
- PyMuPDF (for PDF processing)
- Pillow (for image processing)

## Usage

### Basic Usage

```bash
./run_pdf_ocr.sh --input /path/to/pdf/folder --output /path/to/output/folder
```

### With Markdown Conversion (Default)

```bash
./run_pdf_ocr.sh -i ./input_pdfs -o ./output_results --prompt markdown
```

### With Free OCR Mode

```bash
./run_pdf_ocr.sh -i ./input_pdfs -o ./output_results --prompt free
```

### Direct Python Usage (if already in deepseek-ocr env)

```bash
python process_pdfs.py --input ./input_pdfs --output ./output_results
```

## Arguments

- `--input` or `-i`: Input folder containing PDF files (required)
- `--output` or `-o`: Output folder for OCR results (required)
- `--prompt` or `-p`: Prompt type - `markdown` or `free` (default: markdown)

## Output Structure

```
output_folder/
├── document1/
│   ├── document1_images/        # Temporary images from PDF
│   │   ├── page_1.png
│   │   ├── page_2.png
│   │   └── ...
│   ├── page_1_result.md         # Individual page results
│   ├── page_2_result.md
│   └── document1_full.txt       # Combined results for all pages
├── document2/
│   └── ...
```

## Example

```bash
# Create test folders
mkdir -p input_pdfs output_results

# Put your PDFs in input_pdfs folder
cp /path/to/your/*.pdf input_pdfs/

# Run processing
./run_pdf_ocr.sh -i input_pdfs -o output_results

# Check results
ls -la output_results/
```

## GPU Configuration

The script is configured to use:
- **GPU**: RTX 3090 (PyTorch GPU 0)
- **CUDA_VISIBLE_DEVICES**: '0' (RTX 3090)
- **Precision**: bfloat16 for optimal performance

## Processing Settings

The script uses these optimal settings:
- `base_size`: 1024
- `image_size`: 640
- `crop_mode`: True (Gundam mode - best for documents)
- `save_results`: True
- `test_compress`: True

## Notes

- The model is automatically loaded from: `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR`
- Processing time depends on PDF size and page count
- Each page is processed sequentially to ensure quality
- Temporary images are saved alongside results for reference
