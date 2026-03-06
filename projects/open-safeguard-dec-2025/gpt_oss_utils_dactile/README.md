This directory puts together some utils used in my exploration
of GPT-OSS-Safeguard. This is a partial mirror of 
[DNGros/gptossexp-workspace](https://github.com/DNGros/gptossexp-workspace) 
trying to pull out some of the more generally useful bits 
without as much cruft. However, it's still pretty hacky and might fall
out of date with my main experiments repo if I make more edits in the future.

## Hackathon Contributions

Notes from the hackathon are written up in this post:

[Dev Notes 24: Small Evaluation of GPT-OSS-Safeguard Preferences; Also Prompting Experiments](https://dactile.net/p/dev-notes-24/article.html)

The main points was around exploring the "policy-free" preference 
of the Safeguard model vs the base model. 
This was done by soliciting ratings on how "good" a small set of controversial opinions are. The findings can be summarized as:
- GPT-OSS-Safeguard-20B has measurably different views on how "good" various controversial opinions are. On the little toy collection of 102 controversial opinions rated on a scale from 1 to 5, the two models gave different ratings in ~44% of utterances.
- The GPT-OSS-Safeguard model appears to concentrate more of its ratings of controversial opinions to be the middle indecisive '3' (on a 1 to 5 scale). 72% of its ratings were '3' compared to 55% from the base model.
- I previously had tried to replicate OpenAI's reported results on the ToxicChat dataset. Yacine @ HF had interesting insight at the event around how to prompt the model with the Harmony format as done in the Huggingface Space. However, when I tried after the event with a simple prompt on ToxicChat, the prompt formats did not make much difference. This is perhaps relieving, as it means the simpler approach of passing in the policy into the "system" role might not be much worse than the harmony style. Still more to explore here, and it would be great to hear if anyone else explores/explored replicating results from the paper or looks at comparing prompt format or HF Inference vs Local inference. I'm unsure if local and groq actually return equivalent outputs.

![Plot comparing the data](gpt_oss_utils_dactile/hackathon/likert_distribution_comparison.svg)

## What's in this Path

This path includes code for the hackathon results as well as a potentially
useful framework to build off for testing policies.

### Installation

Setup miniconda / uv / etc for managing the environment. Eg:
```bash
# Activate the environment you are using (or make a new one)
conda create -n gptosoexp python=3.12
conda activate gptossexp
# Install this dir
pip install -e .
```

### Policy Test Framework

We provide a basic framework that we used for iterating on policies.
It's primitive, but the idea is to add a module with a policy and a parse function.
```python
# toy_example_policy.py

POLICY = """\
Output exactly `1` if the user message is toxic, and `0` if it is not
"""

def parse(response: ModelResponse) -> ClassificationResult:
    text = response.response.strip()
    binary_label = (text == "1")
    return ClassificationResult(
        binary_label=binary_label,
        fine_grain_label="1" if binary_label else "0",
        float_label=1.0 if binary_label else 0.0,
        metadata={},
        model_response=response,
        parsed_successfully=(text == "1" or text == "0")
    )
```
See [gpt_oss_utils_dactile/policies](./gpt_oss_utils_dactile/policies) directory for other examples.

Then you can use the classify method.
```python
from gpt_oss_utils_dactile.classify import classify, ClassificationResult
import toy_example_policy
from gpt_oss_utils_dactile.model_predict import ModelResponse, Model, InferenceBackend

response: ClassificationResult = classify(
    text="I hate puppies",
    policy=toy_example_policy, # Just pass in your policy module
    model = Model.GPT_OSS_20B,
    backend = InferenceBackend.API_INJECT_HARMONY,
    #         ^ This is the version that follows the HF space style.
    #           See the post for explaining some of this nuance.
    #           If you instead use InferenceBackend.API it will instead
    #           just pass things into the `system` prompt rather 
    #           than the `developer` channel.
    use_cache: bool = True,
    #                 ^ cache this for rerun later
)
print(response.binary_label)
``` 

### Code for Hackathon Experiment

Entry point there is [gpt_oss_utils_dactile/hackathon/hello_controversial.py](./gpt_oss_utils_dactile/hackathon/hello_controversial.py)
```shell
# Runs all 102 examples with both models and prints some analysis outputs
python -m gpt_oss_utils_dactile.hackathon.hello_controversial
```

Again, it is hacky. The synthesized data there 
is in the [controversial-micro.jsonl](./gpt_oss_utils_dactile/hackathon/controversial-micro.jsonl) file.

### Contact

Feel free to make a discussion and tag @DNGros