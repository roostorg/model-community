"""Minimal vLLM server for gpt-oss-safeguard:20b with Harmony format."""

from fastapi import FastAPI
from pydantic import BaseModel
from openai_harmony import (
    Conversation, Message, Role, SystemContent, DeveloperContent,
    HarmonyEncodingName, load_harmony_encoding,
)
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL = "openai/gpt-oss-safeguard-20b"

app = FastAPI()
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
llm = LLM(model=MODEL, trust_remote_code=True)


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    system_prompt: str | None = None
    developer_prompt: str | None = None


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    # Build harmony conversation
    msgs = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new())]

    if req.developer_prompt:
        msgs.append(Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(req.developer_prompt)
        ))

    for m in req.messages:
        role = {"user": Role.USER, "assistant": Role.ASSISTANT}.get(m["role"], Role.USER)
        msgs.append(Message.from_role_and_content(role, m["content"]))

    conversation = Conversation.from_messages(msgs)

    # Render conversation to token IDs
    prefill_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    # Get stop tokens and configure sampling
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop_token_ids=stop_token_ids,
    )

    # Generate response
    prompt = TokensPrompt(prompt_token_ids=prefill_ids)
    completion_token_ids = llm.generate(prompts=[prompt], sampling_params=sampling_params)[0].outputs[0].token_ids

    # Parse Harmony output channels
    response_messages = encoding.parse_messages_from_completion_tokens(completion_token_ids, Role.ASSISTANT)

    # Extract reasoning and verdict from channels
    reasoning = ""
    verdict = ""
    for msg in response_messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.channel == "final":
            verdict = content.strip()
        else:
            reasoning += content

    # Fallback if no final channel
    if not verdict and response_messages:
        content = response_messages[-1].content
        verdict = content if isinstance(content, str) else str(content)

    return {
        "reasoning": reasoning.strip(),
        "verdict": verdict,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
