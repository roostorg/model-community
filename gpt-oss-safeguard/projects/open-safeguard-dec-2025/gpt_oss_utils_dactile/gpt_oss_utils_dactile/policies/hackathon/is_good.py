from gpt_oss_utils_dactile.classify import ClassificationResult
from gpt_oss_utils_dactile.model_predict import ModelResponse
from gpt_oss_utils_dactile.policies.hackathon.parse_likert import parse_likert

POLICY = """\
## User message labeling

Label how good user message as a value between 1 and 5.

**Output format**: Return a value `1` (not good) to `5` (good). No explanations."""


def parse(response: ModelResponse) -> ClassificationResult:
    return parse_likert(response, min_value=1, max_value=5)
