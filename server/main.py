import asyncio
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.inference_service import (
    OCRService,
    convert_pdf_to_images,
    is_supported_file,
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
)
from server.db import Database


DATA_ROOT = Path("F:/deepseek-ocr-data")
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = DATA_ROOT / "results"
DB_PATH = DATA_ROOT / "db" / "ocr.sqlite"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DeepSeek OCR Server")
ocr_service = OCRService()
db = Database(DB_PATH)


app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _safe_relative_path(filename: str, fallback: str) -> Path:
    """Strip unsafe path parts while preserving folder structure from uploads."""
    parts = [p for p in Path(filename).parts if p not in ("", ".", "..", "")] or [
        fallback
    ]
    return Path(*parts)


async def _write_upload_file(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)


def _relative_to_uploads(path: Path, upload_root: Path) -> str:
    try:
        return str(path.relative_to(upload_root))
    except ValueError:
        return path.name


def _build_combined_markdown(results: list) -> str:
    lines: list[str] = ["# DeepSeek-OCR Results"]
    for entry in results:
        lines.append(f"\n## {entry['source']}")
        if entry["type"] == "pdf":
            for page in entry["pages"]:
                lines.append(f"\n### Page {page['page']}")
                lines.append(str(page["text"]).strip())
        else:
            lines.append(str(entry["text"]).strip())
    return "\n\n".join(lines).strip() + "\n"


async def _require_user(request: Request) -> dict:
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token.")
    token = auth_header.split(" ", 1)[1].strip()
    user = await asyncio.to_thread(db.get_user_by_token, token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token.")
    return user


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="UI is missing.")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/users/new")
async def create_user() -> JSONResponse:
    user = await asyncio.to_thread(db.create_user)
    return JSONResponse(user)


@app.post("/api/ocr")
async def run_ocr(
    request: Request,
    files: List[UploadFile] = File(...),
    prompt: str = Form("markdown"),
) -> JSONResponse:
    user = await _require_user(request)
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    session_id = uuid4().hex
    session_dir = RESULTS_DIR / user["user_id"] / session_id
    uploads_dir = session_dir / "uploads"
    outputs_dir = session_dir / "outputs"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    await asyncio.to_thread(db.create_session, session_id, user["user_id"], prompt)

    saved_paths: list[Path] = []
    for idx, upload in enumerate(files):
        safe_path = _safe_relative_path(upload.filename or "", f"upload_{idx}")
        destination = uploads_dir / safe_path
        await _write_upload_file(upload, destination)
        saved_paths.append(destination)
        await asyncio.to_thread(
            db.add_file_entry,
            session_id,
            "upload",
            str(destination),
            None,
            upload.filename,
            None,
        )

    target_files = [p for p in saved_paths if p.is_file() and is_supported_file(p)]
    if not target_files:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files found. Accepts images {sorted(IMAGE_EXTENSIONS)} or PDFs.",
        )

    results = []
    for file_path in target_files:
        suffix = file_path.suffix.lower()
        source_label = _relative_to_uploads(file_path, uploads_dir)

        if suffix in PDF_EXTENSIONS:
            page_images = convert_pdf_to_images(
                file_path, outputs_dir / f"{file_path.stem}_pages"
            )
            page_results = []
            for page_idx, image_path in enumerate(page_images, 1):
                text = await ocr_service.infer_image(
                    image_path, prompt_key=prompt, output_dir=outputs_dir / file_path.stem
                )
                page_results.append({"page": page_idx, "text": text})
                await asyncio.to_thread(
                    db.add_file_entry,
                    session_id,
                    "pdf_page",
                    str(image_path),
                    str(outputs_dir / file_path.stem),
                    file_path.name,
                    page_idx,
                )
            results.append({"source": source_label, "type": "pdf", "pages": page_results})
        elif suffix in IMAGE_EXTENSIONS:
            text = await ocr_service.infer_image(
                file_path, prompt_key=prompt, output_dir=outputs_dir / file_path.stem
            )
            results.append({"source": source_label, "type": "image", "text": text})
            await asyncio.to_thread(
                db.add_file_entry,
                session_id,
                "image",
                str(file_path),
                str(outputs_dir / file_path.stem),
                file_path.name,
                None,
            )

    combined = _build_combined_markdown(results)
    combined_md = session_dir / "combined.md"
    combined_txt = session_dir / "combined.txt"
    combined_md.write_text(combined, encoding="utf-8")
    combined_txt.write_text(combined, encoding="utf-8")
    await asyncio.to_thread(
        db.finalize_session, session_id, str(combined_md), str(combined_txt)
    )

    return JSONResponse(
        {
            "session_id": session_id,
            "user_id": user["user_id"],
            "combined_markdown": f"/results/{user['user_id']}/{session_id}/{combined_md.name}",
            "combined_text": f"/results/{user['user_id']}/{session_id}/{combined_txt.name}",
            "items": results,
        }
    )


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=10103,
        reload=False,
    )
