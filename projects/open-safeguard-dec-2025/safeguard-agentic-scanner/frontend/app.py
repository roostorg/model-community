import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Ensure repo root (shared package) is on path when launched from Streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

import requests
import streamlit as st

from shared.analyzer import analyze_repo_url, analyze_zip_file, serialize_result


BACKEND_URL = os.getenv("BACKEND_URL")


def _call_backend(url: str, payload: Dict, files: Optional[Dict] = None) -> Dict:
    if not BACKEND_URL:
        raise RuntimeError("BACKEND_URL is not set; running in local mode.")
    endpoint = f"{BACKEND_URL}{url}"
    if files:
        resp = requests.post(endpoint, data=payload, files=files)
    else:
        resp = requests.post(endpoint, json=payload)
    resp.raise_for_status()
    return resp.json()


def _analyze_repo_url(repo_url: str, propose_patch: bool, use_llm: bool, llm_model: str, llm_provider: str):
    if BACKEND_URL:
        return _call_backend(
            "/analyze/url",
            {
                "repo_url": repo_url,
                "propose_patch": propose_patch,
                "use_llm": use_llm,
                "llm_model": llm_model,
                "llm_provider": llm_provider,
            },
        )
    result = analyze_repo_url(
        repo_url,
        propose_patch=propose_patch,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_provider=llm_provider,
    )
    return serialize_result(result)


def _analyze_zip(
    file_bytes: bytes,
    filename: str,
    propose_patch: bool,
    use_llm: bool,
    llm_model: str,
    llm_provider: str,
):
    if BACKEND_URL:
        files = {"file": (filename, file_bytes)}
        return _call_backend(
            "/analyze/upload",
            {
                "propose_patch": str(propose_patch).lower(),
                "use_llm": str(use_llm).lower(),
                "llm_model": llm_model,
                "llm_provider": llm_provider,
            },
            files=files,
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        result = analyze_zip_file(
            tmp_path,
            propose_patch=propose_patch,
            use_llm=use_llm,
            llm_model=llm_model,
            llm_provider=llm_provider,
        )
        return serialize_result(result)
    finally:
        tmp_path.unlink(missing_ok=True)


def render_findings(data: Dict):
    st.subheader("Risk summary")
    if not data.get("findings"):
        st.success("No risky patterns detected by heuristics.")
        return

    summary = data.get("summary", {})
    st.write({k: summary.get(k, 0) for k in ("high", "medium", "low")})

    st.subheader("Findings")
    rows = [
        {
            "severity": f["severity"],
            "rule": f["rule_id"],
            "file": f["file_path"],
            "line": f["line_no"],
            "code": f["code_line"],
            "explanation": f["explanation"],
            "recommendation": f["recommendation"],
        }
        for f in data["findings"]
    ]
    st.dataframe(rows, use_container_width=True)

    if data.get("proposed_patch"):
        st.subheader("Proposed patch")
        st.code(data["proposed_patch"], language="diff")

    if data.get("llm_report"):
        st.subheader("LLM (gpt-oss-safeguard) review")
        st.markdown(data["llm_report"])
    if data.get("llm_patch"):
        st.subheader("LLM patch")
        st.code(data["llm_patch"], language="diff")


def main():
    st.set_page_config(page_title="Safeguard Agentic Scanner", layout="wide")
    st.title("Agentic Code Vulnerability Scanner")
    st.caption(
        "Detects agentic-specific risks (unsafe tools, missing approval, prompt injection, credential leakage) "
        "in a Git repo or uploaded ZIP. Uses heuristics; review manually."
    )

    st.subheader("LLM refinement (optional)")
    use_llm = st.checkbox("Use an LLM to refine findings and patch suggestions", value=False)
    llm_provider = st.selectbox("LLM provider", ["ollama", "huggingface"], index=0)
    default_model = "gpt-oss:20b-cloud" if llm_provider == "ollama" else "openai/gpt-oss-safeguard-20b"
    llm_model = st.text_input(
        "Model name",
        value=default_model,
        help="For Hugging Face, set HUGGINGFACE_API_TOKEN in your env. Recommended: openai/gpt-oss-safeguard-20b.",
    )

    mode = st.tabs(["Repository URL", "Upload ZIP"])

    with mode[0]:
        repo_url = st.text_input("Repository URL")
        propose_patch = st.checkbox("Generate patch suggestions (commented diffs)", value=False, key="url_patch")
        if st.button("Analyze repository", type="primary", use_container_width=False, key="url_btn"):
            if not repo_url:
                st.error("Enter a repository URL.")
            else:
                with st.spinner("Scanning repository..."):
                    try:
                        data = _analyze_repo_url(repo_url, propose_patch, use_llm, llm_model, llm_provider)
                        render_findings(data)
                    except Exception as exc:
                        st.error(f"Scan failed: {exc}")

    with mode[1]:
        uploaded = st.file_uploader("Upload a ZIP archive", type=["zip"])
        propose_patch_zip = st.checkbox("Generate patch suggestions (commented diffs)", value=False, key="zip_patch")
        if st.button("Analyze archive", type="primary", use_container_width=False, key="zip_btn"):
            if not uploaded:
                st.error("Upload a ZIP file.")
            else:
                with st.spinner("Scanning archive..."):
                    try:
                        data = _analyze_zip(
                            uploaded.getbuffer(),
                            uploaded.name,
                            propose_patch_zip,
                            use_llm,
                            llm_model,
                            llm_provider,
                        )
                        render_findings(data)
                    except Exception as exc:
                        st.error(f"Scan failed: {exc}")


if __name__ == "__main__":
    main()
