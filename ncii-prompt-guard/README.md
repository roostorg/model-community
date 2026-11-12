# NCII Prompt Guard

NCII Prompt Guard is a family of lightweight text-classification models from Hugging Face's ML & Society team, designed as prompt guardrails for AI image and video editing apps. The models score editing prompts for Non-Consensual Intimate Imagery (NCII) risk — e.g. "nudification" requests — upstream of the editing model, adding friction against abuse while preserving benign editing functionality. They are 268M-parameter classifiers fine-tuned from [microsoft/harrier-oss-v1-270m](https://huggingface.co/microsoft/harrier-oss-v1-270m), optimized to run on CPU with low latency, and released under the MIT license.

The family includes two versions trained on the same dataset at different label granularities:

- **3-class** ([ncii-prompt-guard-3c](https://huggingface.co/hfmlsoc/ncii-prompt-guard-3c)): severity bands `low` / `review` / `block`; 0.902 block recall at 0.926 block precision on the test split
- **5-class** ([ncii-prompt-guard-5c](https://huggingface.co/hfmlsoc/ncii-prompt-guard-5c)): full label spec `safe` / `suggestive-ambiguous` / `sexualizing-exposure` / `intimate-exposure` / `explicit-sexual-activity`; 0.891 block recall at 0.938 block precision

Both load through the standard `transformers` text-classification pipeline, and deployers set their own operating thresholds on the returned class scores. The models only see the prompt text and cannot adjudicate consent: they are intended to reduce a system's utility for NCII abuse, not to catch individual cases. Training was primarily on English-language data.

## Links

- [NCII Prompt Guard collection on Hugging Face](https://huggingface.co/collections/hfmlsoc/ncii-prompt-guard)
- [Testing Space: compare model behavior at different thresholds](https://huggingface.co/spaces/yjernite/ncii-guard-family-compare)
- [Context article: NCII risk factors, governance, and technical mitigations](https://huggingface.co/datasets/hfmlsoc/ncii-guardrails-article)

### Helpful Resources

- [Training, validation, and test data: NCII Prompt Guard Dataset](https://huggingface.co/datasets/hfmlsoc/ncii-prompt-guard-data) (gated, access on request)
- [Per-group bias and disparate-impact evaluation](https://huggingface.co/hfmlsoc/ncii-prompt-guard-3c#bias-and-disparate-impact) (in the model cards)
- [Embedding atlas of the training and evaluation data](https://huggingface.co/spaces/yjernite/ncii-guard-embedding-atlas)
- [Example deployment on an image-to-video demo Space](https://huggingface.co/spaces/MiniMaxAI/MiniMax-H3-Turbo-Lora/commit/4e2a3d474f8f13a1c16a45f5048bfcc119a762e1)
- [ncii-light-guard-v01](https://huggingface.co/hfmlsoc/ncii-light-guard-v01): previous binary model, superseded by the 3-class/5-class family

