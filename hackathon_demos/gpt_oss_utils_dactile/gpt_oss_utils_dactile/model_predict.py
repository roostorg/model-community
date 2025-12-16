from dataclasses import dataclass
from tqdm import tqdm
import re
from openai_harmony import (
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
)
import hashlib
import json
from pathlib import Path
from huggingface_hub import InferenceClient
from enum import StrEnum
import diskcache
from typing import Literal, Optional, Union, List

# Default batch size for LOCAL backend inference
DEFAULT_BATCH_SIZE = 8

class Model(StrEnum):
    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_safeguard_20B = "openai/gpt-oss-safeguard-20b"


class InferenceBackend(StrEnum):
    """Backend for model inference."""
    API = "api"          # Hugging Face Inference API with standard messages
    API_INJECT_HARMONY = "api_inject_harmony"  # HF API with manually injected Harmony encoding
    LOCAL = "local"      # Local transformers with standard messages
    LOCAL_INJECT_HARMONY = "local_inject_harmony"  # Local with manually injected Harmony encoding


# Cache directory for storing responses
CACHE_DIR = Path(".cache/model_responses")
cache = diskcache.Cache(str(CACHE_DIR))


class LocalModelCache:
    """Singleton cache for local pipeline generators to avoid reloading."""
    _instance = None
    _pipelines = {}
    _current_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_pipeline(self, model_id: str):
        """Get or load text-generation pipeline.

        Only keeps one model in memory at a time to conserve GPU memory.
        If switching models, clears the previous model from GPU.
        """
        # If switching to a different model, clear the old one
        if self._current_model is not None and self._current_model != model_id:
            print(f"Switching models: unloading {self._current_model}")
            self._unload_current_model()

        if model_id not in self._pipelines:
            print(f"Loading model pipeline: {model_id} (this may take a while...)")

            # Lazy import to avoid loading transformers unless needed
            from transformers import pipeline, AutoTokenizer

            # Load tokenizer with left padding for decoder-only models
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.padding_side = 'left'
            
            # Ensure pad token is set (required for batched generation)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self._pipelines[model_id] = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tokenizer,
                torch_dtype="auto",
                device_map="auto",  # Automatically place on available GPUs
            )
            print(f"Pipeline loaded successfully")

        self._current_model = model_id
        return self._pipelines[model_id]

    def _unload_current_model(self):
        """Unload current model and clear GPU memory."""
        if self._current_model and self._current_model in self._pipelines:
            try:
                import torch
                import gc

                # Delete the pipeline
                del self._pipelines[self._current_model]

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Force garbage collection
                gc.collect()

                print(f"Cleared {self._current_model} from GPU memory")
            except Exception as e:
                print(f"Warning: Error clearing model: {e}")

    def clear(self):
        """Clear all cached pipelines to free memory."""
        self._unload_current_model()
        self._pipelines.clear()
        self._current_model = None


@dataclass
class ModelResponse:
    response: str
    reasoning: str
    model: Model
    prompt: str
    system_prompt: str


def _get_cache_key(
    model: Model, 
    prompt: str, 
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    backend: InferenceBackend = InferenceBackend.API,
) -> str:
    """Generate a cache key from the input parameters."""
    cache_data = {
        "model": str(model),
        "prompt": prompt,
        "system_prompt": system_prompt or "",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if backend != InferenceBackend.API:
        # For backwards compatibility only add if new endpoint
        cache_data["backend"] = str(backend)
    cache_string = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_string.encode()).hexdigest()


def clear_cache_by_backend(backend: InferenceBackend = None):
    """
    Clear cached responses, optionally filtered by backend.
    
    Args:
        backend: If specified, only clear cache for this backend.
                 If None, clears entire cache.
    
    Returns:
        int: Number of entries cleared
    """
    if backend is None:
        # Clear entire cache
        count = len(cache)
        cache.clear()
        print(f"Cleared entire cache: {count} entries")
        return count
    
    # Clear only entries for specific backend
    # We need to iterate through cache and check each entry
    keys_to_delete = []
    for key in cache.iterkeys():
        try:
            response = cache.get(key)
            # Check if this is a ModelResponse (it should be)
            if isinstance(response, ModelResponse):
                # We need to reconstruct what backend was used
                # Try to regenerate the key with the specified backend
                test_key = _get_cache_key(
                    model=response.model,
                    prompt=response.prompt,
                    system_prompt=response.system_prompt,
                    backend=backend,
                )
                if test_key == key:
                    keys_to_delete.append(key)
        except Exception:
            # Skip entries that can't be processed
            continue
    
    # Delete the identified keys
    for key in keys_to_delete:
        del cache[key]
    
    print(f"Cleared {len(keys_to_delete)} entries for backend: {backend}")
    return len(keys_to_delete)


def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> list[dict]:
    """
    Build messages list for chat completion (standard format).
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
    
    Returns:
        List of message dicts in chat format
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _build_messages_harmony(
    prompt: str, 
    developer_content: str = None,
    reasoning_effort: Literal["Low", "Medium", "High"] = "Medium"
) -> list[dict]:
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    system_content = SystemContent.new().with_reasoning_effort(reasoning_effort)
    conv_messages = [
        Message.from_role_and_content(Role.SYSTEM, system_content),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(developer_content),
        ),
        Message.from_role_and_content(Role.USER, prompt),
    ]
    print(conv_messages)
    messages = []
    for pre_msg in conv_messages:
        tokens = enc.render(pre_msg)
        prompt = enc.decode(tokens)
        messages.append({
            "role": re.search(r"<\|start\|>(.*?)<\|message\|>", prompt).group(1),
            "content": re.search(r"<\|message\|>(.*?)<\|end\|>", prompt, re.DOTALL).group(1),
        })
    print(messages)
    return messages


def _get_local_model_response(
    model: Model,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    backend: InferenceBackend = InferenceBackend.LOCAL,
) -> ModelResponse:
    """
    Get model response using local transformers pipeline.
    
    Uses the high-level pipeline API which handles chat templates automatically.
    """
    # Get cached pipeline
    model_cache = LocalModelCache()
    generator = model_cache.get_pipeline(str(model))
    
    # Build messages (with or without Harmony encoding)
    if backend == InferenceBackend.LOCAL_INJECT_HARMONY:
        messages = _build_messages_harmony(prompt, system_prompt)
    else:
        messages = _build_messages(prompt, system_prompt)
    
    # Generate response using pipeline
    result = generator(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 1.0,  # pipeline requires temp > 0
        do_sample=(temperature > 0),
    )
    
    # Extract the generated text
    # The pipeline returns the full conversation including the prompt,
    # so we need to get just the assistant's response
    generated_text = result[0]["generated_text"]
    
    # The last message in generated_text should be the assistant's response
    if isinstance(generated_text, list) and len(generated_text) > len(messages):
        assistant_message = generated_text[-1]
        content = assistant_message.get("content", "")
    else:
        # Fallback: use the whole generated text
        content = str(generated_text)

    # Hacky parse the response.
    reasoning, content = content.rsplit("final", 1)
    if reasoning.startswith("analysis"):
        reasoning = reasoning[len("analysis"):]
    
    return ModelResponse(
        response=content,
        reasoning=reasoning,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
    )


def _get_local_model_response_batch(
    model: Model,
    prompts: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    batch_size: int = DEFAULT_BATCH_SIZE,
    backend: InferenceBackend = InferenceBackend.LOCAL,
    use_cache: bool = True,
) -> List[ModelResponse]:
    """
    Get model responses for a batch of prompts using local transformers pipeline.

    Manually chunks prompts into groups for processing to enable real-time progress tracking.
    The pipeline internally batches each chunk with the specified batch_size.
    Each chunk is cached immediately after processing for robustness.

    Args:
        model: Model to use
        prompts: List of prompts to process
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        batch_size: Internal batch size for the pipeline
        backend: Inference backend (for cache key generation)
        use_cache: Whether to cache results
    """
    from tqdm import tqdm

    # Get cached pipeline
    model_cache = LocalModelCache()
    generator = model_cache.get_pipeline(str(model))

    # Build messages for all prompts (with or without Harmony encoding)
    if backend == InferenceBackend.LOCAL_INJECT_HARMONY:
        messages_batch = [_build_messages_harmony(prompt, system_prompt) for prompt in prompts]
    else:
        messages_batch = [_build_messages(prompt, system_prompt) for prompt in prompts]

    # Chunk size: 4x the batch_size for good balance between progress visibility and overhead
    chunk_size = max(batch_size * 4, 1)
    
    backend_desc = "LOCAL+Harmony" if backend == InferenceBackend.LOCAL_INJECT_HARMONY else "LOCAL"
    print(f"Running {backend_desc} inference on {len(prompts)} prompts (batch_size={batch_size}, chunk_size={chunk_size})...")
    
    # Process in chunks to get real-time progress updates and cache each chunk
    all_model_responses = []
    with tqdm(total=len(prompts), desc="LOCAL inference") as pbar:
        for i in range(0, len(messages_batch), chunk_size):
            chunk_messages = messages_batch[i:i + chunk_size]
            chunk_prompts = prompts[i:i + chunk_size]
            
            # Process this chunk (pipeline will internally batch with batch_size)
            chunk_results = generator(
                chunk_messages,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=(temperature > 0),
                batch_size=batch_size,
            )
            
            # Parse results from this chunk into ModelResponse objects
            chunk_model_responses = []
            for prompt, result, messages in zip(chunk_prompts, chunk_results, chunk_messages):
                # Extract the generated text
                generated_text = result[0]["generated_text"]

                # The last message in generated_text should be the assistant's response
                if isinstance(generated_text, list) and len(generated_text) > len(messages):
                    assistant_message = generated_text[-1]
                    content = assistant_message.get("content", "")
                else:
                    # Fallback: use the whole generated text
                    content = str(generated_text)

                # Hacky parse the response
                reasoning, content = content.rsplit("final", 1)
                if reasoning.startswith("analysis"):
                    reasoning = reasoning[len("analysis"):]

                model_response = ModelResponse(
                    response=content,
                    reasoning=reasoning,
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                chunk_model_responses.append(model_response)
                
                # Cache this result immediately
                if use_cache:
                    cache_key = _get_cache_key(model, prompt, system_prompt, temperature, max_tokens, backend)
                    cache.set(cache_key, model_response)
            
            # Collect responses from this chunk
            all_model_responses.extend(chunk_model_responses)
            
            # Update progress bar by the number of prompts processed in this chunk
            pbar.update(len(chunk_messages))

    return all_model_responses


def get_model_response(
    model: Model,
    prompt: Union[str, List[str]],
    instructions: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    use_cache: bool = True,
    backend: InferenceBackend = InferenceBackend.API_INJECT_HARMONY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Union[ModelResponse, List[ModelResponse]]:
    # Detect if batch or single
    is_batch = isinstance(prompt, list)
    prompts = prompt if is_batch else [prompt]
    n = len(prompts)

    # Step 1: Check cache for all prompts
    cache_keys = [
        _get_cache_key(model, p, instructions, temperature, max_tokens, backend)
        for p in prompts
    ]
    cached_results = [cache.get(key) if use_cache else None for key in cache_keys]

    # Step 2: Identify cache misses
    miss_indices = [i for i, cached in enumerate(cached_results) if cached is None]
    miss_prompts = [prompts[i] for i in miss_indices]

    if is_batch and miss_prompts:
        print(f"Cache: {n - len(miss_indices)} hits, {len(miss_indices)} misses")

    if backend == InferenceBackend.LOCAL_INJECT_HARMONY:
        raise NotImplementedError("LOCAL is not supported yet")

    # Step 3: Get responses for cache misses
    miss_responses = []
    if miss_prompts:
        is_local_batch = backend in (InferenceBackend.LOCAL, InferenceBackend.LOCAL_INJECT_HARMONY) and len(miss_prompts) > 1

        if is_local_batch:
            # Batch inference for LOCAL backend - caches each chunk internally
            miss_responses = _get_local_model_response_batch(
                model=model,
                prompts=miss_prompts,
                system_prompt=instructions,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=batch_size,
                backend=backend,
                use_cache=use_cache,
            )
        else:
            # Sequential processing for API backend or single miss
            is_local = backend in (InferenceBackend.LOCAL, InferenceBackend.LOCAL_INJECT_HARMONY)
            desc = "LOCAL inference" if is_local else "API inference"
            for miss_prompt in tqdm(miss_prompts, desc=desc, disable=len(miss_prompts) == 1):
                if is_local:
                    response = _get_local_model_response(
                        model=model,
                        prompt=miss_prompt,
                        system_prompt=instructions,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        backend=backend,
                    )
                else:  # API backend (standard or with injected harmony)
                    client = InferenceClient()
                    
                    # Choose message builder based on backend
                    if backend == InferenceBackend.API_INJECT_HARMONY:
                        messages = _build_messages_harmony(
                            miss_prompt, developer_content=instructions)
                    else:
                        messages = _build_messages(
                            miss_prompt, system_prompt=instructions)

                    params = {}
                    if temperature is not None:
                        params["temperature"] = temperature
                    if max_tokens is not None:
                        params["max_tokens"] = max_tokens

                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **params,
                    )
                    response = ModelResponse(
                        response=completion.choices[0].message.content,
                        reasoning=getattr(completion.choices[0].message, 'reasoning', None) or "",
                        model=model,
                        prompt=miss_prompt,
                        system_prompt=instructions,
                    )

                miss_responses.append(response)

            # Cache the new results (LOCAL batch caches internally, so only cache for API/single)
            for idx, response in zip(miss_indices, miss_responses):
                cache.set(cache_keys[idx], response)

    # Step 4: Merge cached + new in original order
    results = []
    miss_iter = iter(miss_responses)
    for i in range(n):
        if cached_results[i] is not None:
            results.append(cached_results[i])
        else:
            results.append(next(miss_iter))

    # Return single or batch based on input
    return results if is_batch else results[0]


if __name__ == "__main__":
    response = get_model_response(
        model=Model.GPT_OSS_20B,
        prompt="How many 'G's in 'huggingface'?",
        use_cache=False,
    )
    print(response)