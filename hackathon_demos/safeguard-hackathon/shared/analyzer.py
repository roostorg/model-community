import difflib
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Pattern, Tuple

import requests


@dataclass
class Finding:
    rule_id: str
    severity: str
    file_path: str
    line_no: int
    code_line: str
    explanation: str
    recommendation: str


@dataclass
class AnalysisResult:
    source: str
    findings: List[Finding]
    summary: Dict[str, int]
    proposed_patch: Optional[str] = None
    llm_report: Optional[str] = None
    llm_patch: Optional[str] = None


@dataclass
class Rule:
    rule_id: str
    description: str
    severity: str
    recommendation: str
    pattern: Optional[Pattern[str]] = None
    detector: Optional[Callable[[str, List[str]], List[Tuple[int, str]]]] = None


SUPPORTED_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".sh",
    ".md",
    ".yaml",
    ".yml",
    ".json",
}


def _match_regex(pattern: Pattern[str], text: str, lines: List[str]) -> List[Tuple[int, str]]:
    matches: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if pattern.search(line):
            matches.append((idx, line.rstrip()))
    return matches


def _unbounded_tool_loop_detector(filename: str, lines: List[str]) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if re.search(r"while\s+True", line) or re.search(r"for\s+\w+\s+in\s+iter\(int, 1\)", line):
            window = "\n".join(lines[idx - 1 : idx + 4]).lower()
            if "tool" in window or "agent" in window or "action" in window:
                hits.append((idx, line.rstrip()))
    return hits


def _missing_approval_detector(filename: str, lines: List[str]) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if re.search(r"\b(tool|agent)\.run\(", line) or re.search(r"\bexecute_tool", line):
            if "approval" not in line.lower() and "confirm" not in line.lower():
                hits.append((idx, line.rstrip()))
    return hits


def _self_modification_detector(filename: str, lines: List[str]) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if re.search(r"\bgit\s+(commit|push|reset|apply)", line):
            hits.append((idx, line.rstrip()))
        if re.search(r"open\(__file__", line) or re.search(r"\bwrite_text\(", line):
            if ".py" in line or ".js" in line or "__file__" in line:
                hits.append((idx, line.rstrip()))
    return hits


def _prompt_injection_detector(filename: str, lines: List[str]) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if "memory" in line.lower() and ("append" in line or "add" in line) and ("user_input" in line or "message" in line):
            hits.append((idx, line.rstrip()))
        if re.search(r"prompt\s*=.*\+.*input", line.lower()):
            hits.append((idx, line.rstrip()))
    return hits


def _credential_leak_detector(filename: str, lines: List[str]) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if "dotenv" in line.lower() or ".env" in line.lower():
            hits.append((idx, line.rstrip()))
        if re.search(r"os\.environ\.get\([^)]*\)", line) and ("print" in line or "logging." in line):
            hits.append((idx, line.rstrip()))
    return hits


def _overbroad_access_detector(filename: str, lines: List[str]) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if re.search(r"os\.walk\(['\"]/?['\"]", line):
            hits.append((idx, line.rstrip()))
        if re.search(r"requests\.(get|post)\(['\"]http://", line):
            hits.append((idx, line.rstrip()))
    return hits


RULES: List[Rule] = [
    Rule(
        rule_id="unbounded_tool_loop",
        description="Unbounded tool invocation loop without exit or approval",
        severity="high",
        recommendation="Add explicit stop criteria and user approval before looping on tool calls.",
        detector=_unbounded_tool_loop_detector,
    ),
    Rule(
        rule_id="os_system_usage",
        description="Direct shell execution via os.system",
        severity="high",
        recommendation="Replace os.system with a vetted command runner that whitelists commands and checks return codes.",
        pattern=re.compile(r"os\.system\("),
    ),
    Rule(
        rule_id="subprocess_usage",
        description="Broad subprocess invocation",
        severity="high",
        recommendation="Use subprocess.run with explicit command allow-lists and disable shell=True.",
        pattern=re.compile(r"subprocess\.(run|Popen|call|check_output)\("),
    ),
    Rule(
        rule_id="arbitrary_requests",
        description="HTTP requests to arbitrary domains",
        severity="medium",
        recommendation="Restrict outbound network calls to approved domains and add user approval for dynamic URLs.",
        pattern=re.compile(r"requests\.(get|post|put|delete|patch|request)\("),
    ),
    Rule(
        rule_id="missing_user_approval",
        description="Tool invocation without user approval checkpoints",
        severity="medium",
        recommendation="Gate tool executions behind explicit user approval flags or callbacks.",
        detector=_missing_approval_detector,
    ),
    Rule(
        rule_id="autonomous_self_modification",
        description="Code can rewrite or commit itself",
        severity="high",
        recommendation="Disallow self-modification; require human-reviewed patches and restrict git operations.",
        detector=_self_modification_detector,
    ),
    Rule(
        rule_id="prompt_injection_memory",
        description="User input is persisted into memory/prompt without sanitization",
        severity="medium",
        recommendation="Sanitize or filter user-provided content before storing in long-term memory or prompts.",
        detector=_prompt_injection_detector,
    ),
    Rule(
        rule_id="credential_leakage",
        description="Potential credential leakage via .env usage or logging",
        severity="high",
        recommendation="Load secrets through a vault and avoid logging environment variables or .env contents.",
        detector=_credential_leak_detector,
    ),
    Rule(
        rule_id="overbroad_access",
        description="Overbroad filesystem or network access",
        severity="medium",
        recommendation="Constrain file traversal and outbound calls; require scoping to project directories or allow-lists.",
        detector=_overbroad_access_detector,
    ),
    Rule(
        rule_id="unsafe_eval",
        description="Execution of user-controlled input",
        severity="high",
        recommendation="Avoid eval/exec on user data; use safe parsers or sandboxes instead.",
        pattern=re.compile(r"(eval|exec)\s*\([^)]*(input|message|user)", re.IGNORECASE),
    ),
]


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def _scan_file(rule: Rule, file_path: Path) -> List[Finding]:
    findings: List[Finding] = []
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings
    lines = text.splitlines()
    matches: List[Tuple[int, str]] = []
    if rule.pattern:
        matches.extend(_match_regex(rule.pattern, text, lines))
    if rule.detector:
        matches.extend(rule.detector(str(file_path), lines))

    for line_no, line in matches:
        findings.append(
            Finding(
                rule_id=rule.rule_id,
                severity=rule.severity,
                file_path=str(file_path),
                line_no=line_no,
                code_line=line.strip(),
                explanation=rule.description,
                recommendation=rule.recommendation,
            )
        )
    return findings


def _generate_patch(root: Path, findings: List[Finding]) -> Optional[str]:
    by_file: Dict[str, List[Finding]] = {}
    for f in findings:
        by_file.setdefault(f.file_path, []).append(f)

    patches: List[str] = []
    for file_path, file_findings in by_file.items():
        try:
            original_lines = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        updated_lines = original_lines[:]
        comment_prefix = "//" if Path(file_path).suffix in {".js", ".ts", ".tsx"} else "#"
        for finding in file_findings:
            idx = finding.line_no - 1
            if 0 <= idx < len(updated_lines):
                updated_lines[idx] = (
                    f"{comment_prefix} FIXME ({finding.rule_id}): {finding.recommendation} | {updated_lines[idx]}"
                )

        if updated_lines != original_lines:
            diff = "\n".join(
                difflib.unified_diff(
                    original_lines,
                    updated_lines,
                    fromfile=file_path,
                    tofile=file_path,
                    lineterm="",
                )
            )
            patches.append(diff)
    if not patches:
        return None
    return "\n".join(patches)


def _extract_snippet(file_path: str, line_no: int, context: int = 2) -> str:
    try:
        lines = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    start = max(0, line_no - 1 - context)
    end = min(len(lines), line_no + context)
    buf: List[str] = []
    for idx in range(start, end):
        prefix = ">" if idx == line_no - 1 else " "
        buf.append(f"{prefix}{idx + 1}: {lines[idx].rstrip()}")
    return "\n".join(buf)


def _format_findings_for_llm(findings: List[Finding], limit: int = 20) -> str:
    chunks: List[str] = []
    for finding in findings[:limit]:
        snippet = _extract_snippet(finding.file_path, finding.line_no)
        chunks.append(
            "\n".join(
                [
                    f"[{finding.severity}] {finding.rule_id} @ {finding.file_path}:{finding.line_no}",
                    f"line: {finding.code_line}",
                    f"snippet:\n{snippet}" if snippet else "snippet: (not available)",
                    f"why: {finding.explanation}",
                    f"recommend: {finding.recommendation}",
                ]
            )
        )
    return "\n\n".join(chunks)


def _extract_diff_block(text: str) -> Optional[str]:
    fenced = re.search(r"```diff\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    unified = re.search(r"(?m)^--- .*?\n\+\+\+ .*", text, re.DOTALL)
    if unified:
        return text[unified.start() :].strip()
    return None


def _run_hf_review(prompt: str, model: str) -> Tuple[Optional[str], Optional[str]]:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set for Hugging Face inference.")

    endpoint = f"https://router.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Hugging Face inference error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    content: Optional[str] = None
    if isinstance(data, list) and data:
        content = data[0].get("generated_text")
    elif isinstance(data, dict):
        content = data.get("generated_text") or data.get("answer")

    if not content:
        raise RuntimeError(f"Unexpected Hugging Face response: {data}")

    patch = _extract_diff_block(content)
    return content.strip(), patch


def run_llm_review(
    findings: List[Finding],
    source: str,
    model: str = "gpt-oss-safeguard",
    provider: str = "ollama",
) -> Tuple[Optional[str], Optional[str]]:
    prompt = (
        "You are reviewing an agentic codebase for security and safety issues. "
        "Focus on tool loops, subprocess/os.system, arbitrary requests, missing approval gates, "
        "self-modification, prompt injection, and credential leakage. "
        "Given the heuristic findings below, provide clearer explanations and propose a minimal unified diff patch. "
        "Return two sections: 'ANALYSIS:' and 'PATCH:'. If no patch, set PATCH: none.\n\n"
        f"Source: {source}\n\nFindings:\n{_format_findings_for_llm(findings)}"
    )

    if provider == "huggingface":
        return _run_hf_review(prompt, model=model)
    if provider != "ollama":
        raise RuntimeError(f"Unsupported LLM provider: {provider}")

    # Default: Ollama
    try:
        import ollama
    except Exception as exc:
        raise RuntimeError("ollama package is required for LLM enrichment") from exc

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    content = ""
    if isinstance(response, dict):
        content = response.get("message", {}).get("content", "")
    else:
        content = str(response)
    if not content:
        return None, None

    patch = _extract_diff_block(content)
    return content.strip(), patch


def analyze_path(
    source: str,
    path: Path,
    propose_patch: bool = False,
    use_llm: bool = False,
    llm_model: str = "gpt-oss:20b-cloud",
    llm_provider: str = "ollama",
) -> AnalysisResult:
    findings: List[Finding] = []
    for file in _iter_files(path):
        for rule in RULES:
            findings.extend(_scan_file(rule, file))

    summary: Dict[str, int] = {}
    for f in findings:
        summary[f.severity] = summary.get(f.severity, 0) + 1

    patch = _generate_patch(path, findings) if propose_patch else None
    llm_report: Optional[str] = None
    llm_patch: Optional[str] = None
    if use_llm and findings:
        try:
            llm_report, llm_patch = run_llm_review(
                findings,
                source=source,
                model=llm_model,
                provider=llm_provider,
            )
        except Exception as exc:
            llm_report = f"LLM review failed: {exc}"

    return AnalysisResult(
        source=source,
        findings=findings,
        summary=summary,
        proposed_patch=patch,
        llm_report=llm_report,
        llm_patch=llm_patch,
    )


def _clone_repo(repo_url: str, checkout_dir: Path) -> Path:
    repo_dir = checkout_dir / "repo"
    subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], stdout=subprocess.DEVNULL)
    return repo_dir


def _extract_zip(zip_file: Path, target_dir: Path) -> Path:
    import zipfile

    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(target_dir)
    return target_dir


def analyze_repo_url(
    repo_url: str,
    propose_patch: bool = False,
    use_llm: bool = False,
    llm_model: str = "gpt-oss:20b-cloud",
    llm_provider: str = "ollama",
) -> AnalysisResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repo_path = _clone_repo(repo_url, root)
        return analyze_path(
            source=repo_url,
            path=repo_path,
            propose_patch=propose_patch,
            use_llm=use_llm,
            llm_model=llm_model,
            llm_provider=llm_provider,
        )


def analyze_zip_file(
    zip_file: Path,
    propose_patch: bool = False,
    use_llm: bool = False,
    llm_model: str = "gpt-oss:20b-cloud",
    llm_provider: str = "ollama",
) -> AnalysisResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        extracted = _extract_zip(zip_file, root)
        return analyze_path(
            source=str(zip_file),
            path=Path(extracted),
            propose_patch=propose_patch,
            use_llm=use_llm,
            llm_model=llm_model,
            llm_provider=llm_provider,
        )


def serialize_result(result: AnalysisResult) -> Dict:
    return {
        "source": result.source,
        "summary": result.summary,
        "findings": [asdict(f) for f in result.findings],
        "proposed_patch": result.proposed_patch,
        "llm_report": result.llm_report,
        "llm_patch": result.llm_patch,
    }
