Some sample policies for GPT-OSS-Safeguard.

Policies implement the protocol:
```python
class PolicyModule(Protocol):
    POLICY: str
    def parse(self, response: "ModelResponse") -> "ClassificationResult": ...
```
Altertnatively you can ommit POLICY. In which case you must have 
.md file with the same name which is the policy.

## Policies

- `spam.md` A sample spam classification from [OpenAI cookbook](https://cookbook.openai.com/articles/gpt-oss-safeguard-guide)
- `toxic_chat_claude_1.md` A sample policy for label [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) quickly authored by claude. It gives 0 and 1 binary labels.
- `toxic_simple.py` A simple prompt just asking if toxic or a jailbreak with 0 or 1 labels.
- `toxic_known_dataset.py` A toxicchat label built that exploits the model probably already knows what the dataset is.
- hackathon. a likert scoring for hackathon.