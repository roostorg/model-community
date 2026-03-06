from dataclasses import dataclass
import re
from pathlib import Path
from typing import Protocol, Union, List, runtime_checkable
from gpt_oss_utils_dactile.model_predict import ModelResponse, get_model_response, Model, InferenceBackend, DEFAULT_BATCH_SIZE


@dataclass
class ClassificationResult:
    binary_label: bool | None # True = violates policy, False = ok, None if not applicable
    fine_grain_label: dict | str | None
    float_label: float | None
    metadata: dict
    model_response: ModelResponse
    parsed_successfully: bool  # True if parser successfully extracted meaningful information


@runtime_checkable
class PolicyModuleBase(Protocol):
    def parse(self, response: "ModelResponse") -> "ClassificationResult": ...


@runtime_checkable
class PolicyModule(PolicyModuleBase, Protocol):
    """Policy module with inline POLICY string and parse function."""
    POLICY: str


def _default_parse(response: ModelResponse) -> ClassificationResult:
    """Default parser when policy module doesn't provide one."""
    text = response.response
    
    # Extract binary label - INVALID means violates (True)
    binary_label = "INVALID" in text.upper()
    
    # Try to extract fine-grain labels
    fine_grain = None
    
    # Look for patterns like D-SP2, R-SP0, SP3, etc.
    sp_matches = re.findall(r'[DR]-?SP\d+(?:\.[a-z])?', text, re.IGNORECASE)
    if sp_matches:
        # Check if we have both D- and R- labels
        d_labels = [m for m in sp_matches if m.upper().startswith('D')]
        r_labels = [m for m in sp_matches if m.upper().startswith('R')]
        if d_labels or r_labels:
            fine_grain = {
                "depiction": d_labels[0] if d_labels else None,
                "request": r_labels[0] if r_labels else None
            }
        else:
            fine_grain = sp_matches[0] if sp_matches else None
    
    # Parsing was successful if we found INVALID keyword or SP labels
    parsed_successfully = binary_label or (fine_grain is not None)
    
    metadata = {}
    
    return ClassificationResult(
        binary_label=binary_label,
        fine_grain_label=fine_grain,
        float_label=1.0 if binary_label else 0.0,
        metadata=metadata,
        model_response=response,
        parsed_successfully=parsed_successfully
    )


def classify(
    text: Union[str, List[str]],
    policy_module: Union[PolicyModule, PolicyModuleBase],
    model: Model = Model.GPT_OSS_20B,
    backend: InferenceBackend = InferenceBackend.API_INJECT_HARMONY,
    use_cache: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Union[ClassificationResult, List[ClassificationResult]]:
    """
    Classify text using a policy module.

    Accepts either a single text or a list of texts for batch processing.

    Args:
        text: Text to classify (single string or list of strings)
        policy_module: Python module with POLICY string or put .md file at same path
        model: Model to use for classification
        backend: Inference backend (API or LOCAL)
        use_cache: Whether to use cache
        batch_size: Batch size for LOCAL backend

    Returns:
        ClassificationResult (single) or List[ClassificationResult] (batch)
    """
    # Detect if batch or single
    is_batch = isinstance(text, list)
    texts = text if is_batch else [text]

    # Get system prompt: prefer POLICY attribute, fallback to .md file
    if hasattr(policy_module, 'POLICY'):
        system_prompt = policy_module.POLICY
    else:
        # Legacy: load the markdown file from the policy module's directory
        module_path = Path(policy_module.__file__).parent
        module_name = policy_module.__name__.split('.')[-1]
        md_path = module_path / f"{module_name}.md"

        if not md_path.exists():
            raise FileNotFoundError(
                f"Policy module must have a POLICY attribute or a corresponding .md file. "
                f"Neither found for {policy_module.__name__}"
            )

        system_prompt = md_path.read_text()

    # Get model responses (batched if list)
    # Since texts is always a list, get_model_response will return a list
    responses = get_model_response(
        model=model,
        prompt=texts,
        instructions=system_prompt,
        backend=backend,
        use_cache=use_cache,
        batch_size=batch_size,
    )

    # responses is already a list since we always pass texts as a list

    # Use custom parser if available, otherwise default
    parser = policy_module.parse if hasattr(policy_module, 'parse') else _default_parse
    results = [parser(response) for response in responses]

    # Return single or batch based on input
    return results if is_batch else results[0]

