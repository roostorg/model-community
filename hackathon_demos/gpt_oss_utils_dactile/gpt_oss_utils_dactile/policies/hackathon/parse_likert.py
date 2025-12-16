import re
from gpt_oss_utils_dactile.classify import ClassificationResult
from gpt_oss_utils_dactile.model_predict import ModelResponse


def parse_likert(
    response: ModelResponse,
    min_value: int = 1,
    max_value: int = 5,
) -> ClassificationResult:
    """Parse likert scale response.
    
    Searches for a number in the text and binarizes it by the midpoint
    of the min/max range. Values > midpoint are True (violates policy).
    """
    text = response.response.strip()
    
    # Search for a number in the text
    match = re.search(r'\b(\d+)\b', text)
    
    if match:
        value = int(match.group(1))
        # Clamp to valid range
        value = max(min_value, min(max_value, value))
        midpoint = (min_value + max_value) / 2
        binary_label = value > midpoint
        float_label = (value - min_value) / (max_value - min_value)  # Normalize to 0-1
        parsed_successfully = True
    else:
        # No number found
        value = None
        binary_label = False
        float_label = None
        parsed_successfully = False
    
    return ClassificationResult(
        binary_label=binary_label,
        fine_grain_label=value,
        float_label=float_label,
        metadata={"raw_value": value, "min": min_value, "max": max_value},
        model_response=response,
        parsed_successfully=parsed_successfully,
    )