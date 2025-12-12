import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# Ensure repo root (shared package) is on path when launched via uvicorn
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from shared.analyzer import analyze_repo_url, analyze_zip_file, serialize_result


class AnalyzeUrlRequest(BaseModel):
    repo_url: HttpUrl
    propose_patch: bool = False
    use_llm: bool = False
    llm_model: str = "gpt-oss:20b-cloud"
    llm_provider: str = "ollama"


app = FastAPI(title="Safeguard Agentic Scanner", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze/url")
def analyze_url(payload: AnalyzeUrlRequest):
    result = analyze_repo_url(
        str(payload.repo_url),
        propose_patch=payload.propose_patch,
        use_llm=payload.use_llm,
        llm_model=payload.llm_model,
        llm_provider=payload.llm_provider,
    )
    return serialize_result(result)


@app.post("/analyze/upload")
def analyze_upload(
    file: UploadFile = File(...),
    propose_patch: bool = Form(False),
    use_llm: bool = Form(False),
    llm_model: str = Form("gpt-oss:20b-cloud"),
    llm_provider: str = Form("ollama"),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "").suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        result = analyze_zip_file(
            tmp_path,
            propose_patch=propose_patch,
            use_llm=use_llm,
            llm_model=llm_model,
            llm_provider=llm_provider,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return serialize_result(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
