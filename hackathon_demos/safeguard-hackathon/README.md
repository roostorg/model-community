# Safeguard Agentic Scanner

Fast heuristic scanner for agentic repos. Given a Git URL or uploaded ZIP it:
- Detects dangerous patterns (unbounded tool loops, unsafe subprocess/requests usage, missing approval gates, prompt injection risks, credential leakage, overbroad access, self-modification, unsafe `eval/exec`).
- Explains why each finding is unsafe and how to remediate.
- Optionally emits a commented patch diff to flag risky lines inline.

Built with FastAPI (backend) and Streamlit (frontend). Analysis logic lives in `shared/analyzer.py` so both layers share the same rules.

## Quickstart
1) Create env & install deps
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
uv pip install -r requirements.txt
```

2) Run the backend
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

3) Run the frontend (set BACKEND_URL so Streamlit calls the API; without it, Streamlit runs the analyzer locally)
```bash
set BACKEND_URL=http://localhost:8000
streamlit run frontend/app.py
```

Hugging Face Spaces (Streamlit)
- Keep `app.py` at repo root as the Space entrypoint.
- Add `HUGGINGFACE_API_TOKEN` (for HF provider) and/or `OLLAMA_HOST` as Space secrets if needed.
- Deploy as a Streamlit Space; it will run `streamlit run app.py` by default.

Optional: start Ollama with the safeguard model for LLM refinement
```bash
ollama run gpt-oss:20b-cloud  # downloads model on first run
# If your Ollama host is remote, export OLLAMA_HOST=http://host:11434
```

Optional: use Hugging Face inference (e.g., `openai/gpt-oss-safeguard-20b`)
```bash
HUGGINGFACE_API_TOKEN=your_token_here
# place in .env or export in your shell
# Choose provider=huggingface and model=openai/gpt-oss-safeguard-20b in UI/API
```

## API examples
- Health: `curl http://localhost:8000/health`
- Analyze repo URL:
```bash
curl -X POST http://localhost:8000/analyze/url \
  -H "Content-Type: application/json" \
  -d "{\"repo_url\":\"https://github.com/example/agentic-repo\",\"propose_patch\":true,\"use_llm\":true,\"llm_model\":\"gpt-oss:20b-cloud\",\"llm_provider\":\"ollama\"}"
```
- Analyze uploaded ZIP:
```bash
curl -X POST http://localhost:8000/analyze/upload \
  -F "file=@repo.zip" \
  -F "propose_patch=true" \
  -F "use_llm=true" \
  -F "llm_model=gpt-oss:20b-cloud" \
  -F "llm_provider=ollama"
```

## How detection works
- Rules live in `shared/analyzer.py` as regex/custom detectors per risk category.
- Supported file types: `.py`, `.js`, `.ts`, `.tsx`, `.sh`, `.md`, `.yaml`, `.yml`, `.json`.
- Findings include file, line, severity, explanation, and a remediation tip.
- Patch suggestions (optional) prepend a `#`/`//` FIXME comment to risky lines and output a unified diff without modifying the repo.
- LLM refinement (optional) sends the findings and code snippets to `gpt-oss-safeguard` via Ollama or `openai/gpt-oss-safeguard-20b` via Hugging Face and returns a richer analysis plus a model-suggested diff block if present.

## Extending
- Add or tune rules in `shared/analyzer.py` (`RULES` list and detectors).
- Swap the LLM provider/model via UI/API. For Hugging Face, set `HUGGINGFACE_API_TOKEN` and use `openai/gpt-oss-safeguard-20b`. Additional providers (Groq/OpenAI) can be added by extending `run_llm_review`.
- Tighten network/domain allow-lists or approval semantics by enriching `_missing_approval_detector` or adding config-driven policies.

## Notes
- Scanner is heuristic and can surface false positives/negatives. Use it as triage; review manually.
- Git cloning uses `git clone --depth 1`; ensure the host is reachable from where you run the backend.
