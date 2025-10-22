from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import argparse

# Set CUDA device to RTX 3090 (GPU 0 in PyTorch)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def convert_pdf_to_images(pdf_path, output_folder):
    """Convert PDF pages to images"""
    doc = fitz.open(pdf_path)
    image_paths = []
    
    pdf_name = Path(pdf_path).stem
    img_folder = Path(output_folder) / f"{pdf_name}_images"
    img_folder.mkdir(parents=True, exist_ok=True)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to an image (higher DPI for better quality)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_path = img_folder / f"page_{page_num + 1}.png"
        pix.save(str(img_path))
        image_paths.append(str(img_path))
    
    doc.close()
    return image_paths

def process_pdfs(input_folder, output_folder, model, tokenizer, prompt_type='markdown'):
    """Process all PDFs in the input folder"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define prompts
    prompts = {
        'markdown': "<image>\n<|grounding|>Convert the document to markdown. ",
        'free': "<image>\nFree OCR. "
    }
    prompt = prompts.get(prompt_type, prompts['markdown'])
    
    # Check if input is a single PDF file or a folder
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
    else:
        print(f"Error: {input_folder} is neither a PDF file nor a directory")
        return
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_file.name}")
        print(f"{'='*60}")
        
        try:
            # Convert PDF to images
            print("Converting PDF pages to images...")
            image_paths = convert_pdf_to_images(str(pdf_file), output_path)
            print(f"Created {len(image_paths)} images from PDF")
            
            # Create output directory for this PDF
            pdf_output_dir = output_path / pdf_file.stem
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each page
            all_results = []
            for idx, image_path in enumerate(image_paths, 1):
                print(f"\nProcessing page {idx}/{len(image_paths)}...")
                
                # Run OCR on this single page
                result = model.infer(
                    tokenizer, 
                    prompt=prompt, 
                    image_file=image_path, 
                    output_path=str(pdf_output_dir / f"page_{idx}"), 
                    base_size=1024, 
                    image_size=640, 
                    crop_mode=True, 
                    save_results=True, 
                    test_compress=True
                )
                
                all_results.append({
                    'page': idx,
                    'result': result
                })
                print(f"✓ Page {idx} processed successfully")
            
            # Save combined results
            combined_output_file = pdf_output_dir / f"{pdf_file.stem}_combined.mmd"
            with open(combined_output_file, 'w', encoding='utf-8') as f:
                for idx, res in enumerate(all_results, 1):
                    if idx > 1:
                        f.write(f"\n\n<--- Page {idx} Split --->\n\n")
                    f.write(str(res['result']))
            
            print(f"\n✓ PDF processed successfully!")
            print(f"  Output directory: {pdf_output_dir}")
            print(f"  Combined results: {combined_output_file}")
            print(f"  Total pages processed: {len(all_results)}")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Process PDFs with DeepSeek-OCR')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Input PDF file or folder containing PDF files')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder for OCR results')
    parser.add_argument('--prompt', '-p', type=str, default='markdown',
                        choices=['markdown', 'free'],
                        help='Prompt type: markdown or free (default: markdown)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DeepSeek-OCR PDF Processor")
    print("="*60)
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"Prompt type: {args.prompt}")
    print("="*60)
    
    # Load model
    print("\nLoading tokenizer...")
    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("Loading model (this may take a while)...")
    model = AutoModel.from_pretrained(
        model_name, 
        _attn_implementation='flash_attention_2', 
        trust_remote_code=True, 
        use_safetensors=True
    )
    
    print("Loading model to GPU...")
    model = model.eval().cuda().to(torch.bfloat16)
    
    print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Process PDFs
    print("\nStarting PDF processing...\n")
    process_pdfs(args.input, args.output, model, tokenizer, args.prompt)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
