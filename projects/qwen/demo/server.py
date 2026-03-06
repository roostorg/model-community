"""Minimal vLLM server for Qwen3Guard."""

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3Guard-Gen-8B"

app = FastAPI()
llm = LLM(model=MODEL, trust_remote_code=True)


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    system_prompt: str | None = None


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    # Build messages list
    msgs = []

    if req.system_prompt:
        msgs.append({"role": "system", "content": req.system_prompt})

    for m in req.messages:
        msgs.append({"role": m["role"], "content": m["content"]})

    # Generate using vLLM's chat interface
    output = llm.chat(
        msgs,
        SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        ),
    )[0].outputs[0].text

    return {"choices": [{"message": {"role": "assistant", "content": output}}]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
