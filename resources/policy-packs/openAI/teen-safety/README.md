# Teen Safety Policy Pack

<p align="center">
  <strong>Download <a href="https://huggingface.co/openai/gpt-oss-safeguard-120b">gpt-oss-safeguard-120b</a> and <a href="https://huggingface.co/openai/gpt-oss-safeguard-20b">gpt-oss-safeguard-20b</a> on Hugging Face</strong>
</p>
<p align="center">
<a href="https://huggingface.co/spaces/openai/gpt-oss-safeguard-20b"><strong>Try gpt-oss-safeguard</strong></a> ·
  <a href="https://cookbook.openai.com/articles/gpt-oss-safeguard-guide"><strong>Guide</strong></a> ·
  <a href="https://arxiv.org/abs/2508.10925"><strong>Model card</strong></a> ·
  <a href="https://openai.com/index/introducing-gpt-oss-safeguard/"><strong>OpenAI blog</strong></a> 
</p>

The Teen Safety Policy Pack is a set of prompt-based safety policies designed to create age-appropriate protections for teens.

These policies are structured as prompts that can be directly used with [`gpt-oss-safeguard`](https://github.com/openai/gpt-oss-safeguard), enabling developers to turn safety requirements into usable classifiers for real-world systems.

Use of this repository is subject to the [usage policy](./USAGE_POLICY) and the [Apache 2.0 license](./LICENSE).

## How to use the policies

1. Pick the policy that best matches the teen-safety risk you want to evaluate from [`example_policies/`](./example_policies/). Each policy lives in its own folder and the prompt text lives in `policy.md`.
2. Pass that policy prompt to `gpt-oss-safeguard` together with the content you want to classify. The model should use the policy labels and examples in the prompt as the decision framework.
3. Review the model output against the labels defined in the policy and map them into your product workflow, such as real-time filtering, offline review, triage, or monitoring.
4. Adapt the prompt to your product, audience, and operational context. These policies are intended to be starting points, not fixed rulesets.
5. Run the matching validation set in [`datasets/`](./datasets/) before shipping prompt changes so you can measure how your edits affect performance.

## Initial policy areas

The initial release covers:

- Graphic violent content (`graphic-violent-content`)
- Graphic sexual content (`graphic-sexual-content`)
- Harmful body ideals and behaviors (`harmful-body-ideals`)
- Dangerous activities and challenges (`dangerous-content`)
- Dangerous or inappropriate roleplay (`dangerous-roleplay`)
- Age-restricted goods and services (`age-restricted-goods-and-services`)

These policies can be used for real-time content filtering, as well as offline analysis of user-generated content.

By structuring policies as prompts, developers can more easily integrate them into existing workflows, adapt them to their use cases, and iterate over time.

## Validation datasets

Policy validation datasets live in [`datasets/`](./datasets/) as policy-specific CSV files. The filenames mirror the policy slugs in [`example_policies/`](./example_policies/), making it straightforward to pair a prompt with its evaluation set during prompt iteration and regression testing.

## How to contribute

We are releasing these policies as open source through the [ROOST Model Community](https://github.com/roostorg/open-models) to encourage collaboration and iteration. To contribute, provide feedback, or share additional teen safety policies, visit the RMC GitHub repository.

## Limitations

The policies are intended as a starting point, not as a comprehensive or final definition or guarantee of teen safety. Each application has unique risks, audiences, and contexts, and developers are best positioned to understand the risks that their products and AI integrations may present. We strongly encourage developers to adapt and extend these policies based on their specific needs and combine them with other safeguards.
