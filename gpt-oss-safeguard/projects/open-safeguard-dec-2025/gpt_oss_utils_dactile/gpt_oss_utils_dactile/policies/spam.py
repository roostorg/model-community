import re
from gpt_oss_utils_dactile.model_predict import ModelResponse
from gpt_oss_utils_dactile.classify import ClassificationResult


def parse(response: ModelResponse) -> ClassificationResult:
    """Parse spam policy response with D-SP# and R-SP# labels."""
    text = response.response
    
    # Extract D-SP# (depiction) and R-SP# (request) labels
    d_match = re.search(r'D-SP(\d+)(?:\.([a-z]))?', text, re.IGNORECASE)
    r_match = re.search(r'R-SP(\d+)(?:\.([a-z]))?', text, re.IGNORECASE)
    
    fine_grain = {
        "depiction": d_match.group(0).upper() if d_match else None,
        "request": r_match.group(0).upper() if r_match else None,
    }
    
    # Binary label: INVALID if explicitly stated, OR if D-SP level is 2+ (spam detected)
    binary_label = "INVALID" in text.upper()
    if not binary_label and d_match:
        # D-SP0 = not spam, D-SP2+ = spam
        d_level = int(d_match.group(1))
        binary_label = d_level >= 2
    
    # Parsing was successful if we found INVALID keyword or at least one SP label
    parsed_successfully = "INVALID" in text.upper() or d_match is not None or r_match is not None
    
    metadata = {}
    
    return ClassificationResult(
        binary_label=binary_label,
        fine_grain_label=fine_grain,
        metadata=metadata,
        model_response=response,
        parsed_successfully=parsed_successfully
    )

