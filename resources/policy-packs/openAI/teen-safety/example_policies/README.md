# Example Policies

Each policy lives in its own folder with a `policy.md` file containing the prompt text to provide to `gpt-oss-safeguard`.

This mirrors the per-policy layout used in [`openai/gpt-oss-safeguard/example_policies/spam`](https://github.com/openai/gpt-oss-safeguard/tree/main/example_policies/spam), but stores the prompt assets in Markdown.

Matching validation datasets live in [`../datasets/`](../datasets/) as policy-specific CSV files named after the same policy slugs.

Initial policies:

- [harmful-body-ideals](./harmful-body-ideals/policy.md)
- [graphic-violent-content](./graphic-violent-content/policy.md)
- [graphic-sexual-content](./graphic-sexual-content/policy.md)
- [dangerous-content](./dangerous-content/policy.md)
- [age-restricted-goods-and-services](./age-restricted-goods-and-services/policy.md)
- [dangerous-roleplay](./dangerous-roleplay/policy.md)
