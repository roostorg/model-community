import re
from ..model_predict import ModelResponse
from ..classify import ClassificationResult


def parse(response: ModelResponse) -> ClassificationResult:
    """Parse toxic_chat_claude_1 policy response with binary 0/1 labels."""
    text = response.response.strip()
    
    # The policy expects exactly "0" (non-toxic) or "1" (toxic)
    # Look for standalone 0 or 1 in the response
    match = re.search(r'\b([01])\b', text)
    
    if match:
        label_str = match.group(1)
        binary_label = (label_str == "1")  # True if toxic (1), False if non-toxic (0)
        parsed_successfully = True  # Found explicit 0/1 label
    else:
        # Fallback: check for keywords if no clear 0/1 found
        text_upper = text.upper()
        if "TOXIC" in text_upper or "OFFENSIVE" in text_upper:
            binary_label = True
            parsed_successfully = True  # Found keyword indicating toxic
        elif "NON-TOXIC" in text_upper or "NOT OFFENSIVE" in text_upper:
            binary_label = False
            parsed_successfully = True  # Found keyword indicating non-toxic
        else:
            # Default to False (non-toxic) if unclear
            binary_label = False
            parsed_successfully = False  # No clear label found
    
    # Fine-grain label is just the string label for this policy
    fine_grain = "1" if binary_label else "0"
    
    metadata = {
        "raw_response": text
    }
    
    return ClassificationResult(
        binary_label=binary_label,
        fine_grain_label=fine_grain,
        float_label=1.0 if binary_label else 0.0,
        metadata=metadata,
        model_response=response,
        parsed_successfully=parsed_successfully
    )

