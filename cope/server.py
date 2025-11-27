"""Minimal vLLM server for cope-a-9b (Gemma-2-9b + LoRA adapter)."""

import os
# Gemma-2 uses softcapping - requires FLASHINFER backend
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

BASE_MODEL = "google/gemma-2-9b"
ADAPTER_MODEL = "zentropi-ai/cope-a-9b"

app = FastAPI()
llm = LLM(
    model=BASE_MODEL,
    enable_lora=True,
    max_lora_rank=64,
    trust_remote_code=True,
)


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    policy: str | None = None


def build_cope_prompt(policy: str, content: str) -> str:
    """Build the CoPE prompt format with INSTRUCTIONS, POLICY, CONTENT, ANSWER sections."""
    return f"""### INSTRUCTIONS
Examine the POLICY against the CONTENT and respond with the appropriate label.

### POLICY
{policy}

### CONTENT
{content}

### ANSWER
"""


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    # Extract user message content
    content = ""
    for m in req.messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            break

    if not req.policy:
        return {"error": "Policy is required", "reasoning": "", "verdict": ""}

    # Build CoPE prompt
    prompt = build_cope_prompt(req.policy, content)

    # Configure sampling for deterministic binary output
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    # Create LoRA request for the adapter
    lora_request = LoRARequest("cope-adapter", 1, ADAPTER_MODEL)

    # Generate response with LoRA adapter
    outputs = llm.generate(
        prompts=[prompt],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    # Parse output (expects 0 or 1)
    raw_output = outputs[0].outputs[0].text.strip()

    # Convert binary to verdict
    if raw_output.startswith("1"):
        verdict = "Violating"
    elif raw_output.startswith("0"):
        verdict = "Non-violating"
    else:
        verdict = raw_output

    return {
        "reasoning": "",
        "verdict": verdict,
        "raw_output": raw_output,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
