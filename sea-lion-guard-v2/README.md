# SEA-LION-Guard-v2

SEA-LION-Guard is a collection of safety-focused LLMs from AI Singapore, designed specifically for the Southeast Asia (SEA) region. They support Burmese, English, Indonesian, Filipino, Malay, Tamil, Thai and Vietnamese, and are released under the MIT license.

SEA-LION-Guard-v2 is a decision model: it produces structured probabilistic judgments rather than free-form text. It is fine-tuned to answer three core questions:

- `content_moderation`: probability scores (0.0–1.0) for `safe`, `sensitive` and `unsafe`
- `rule_violation`: the probability that the input violates the specified rules
- `rule_applicability`: the probability that the specified rules apply to the input

The earlier SEA-Guard series (now called SEA-LION-Guard v1) was designed only for content moderation. SEA-LION-Guard-v2 supports both **Content Moderation** and **Action Governance**, and adds:

- **Multi-turn conversation evaluation**: evaluates safety across multi-turn interactions
- **Bring Your Own Policy (BYOP)**: lets users define and enforce their own safety rules and policies
- **Tool interaction evaluation**: evaluates tool calls and tool responses for safe, policy-compliant agent behavior

## Links

- [Download SEA-LION-Guard-v2 on HuggingFace](https://huggingface.co/collections/aisingapore/sea-lion-guard-v2)
- [SEA-LION-Guard-v2 Announcement](https://sea-lion.ai/blog/safeguarding-regional-agents-announcing-gemma-sea-lion-guard-v2-31b-e4b-e2b/)
- [SEA-LION-Guard-v2 Playground](https://playground.sea-lion.ai/sea-guard)

### Helpful Resources

- [Training data: SEA-Safeguard-Train-Cultural-v3](https://huggingface.co/datasets/aisingapore/SEA-Safeguard-Train-Cultural-v3) (gated, access on request)
- [SEA-Guard paper (SEA-LION-Guard-v1)](https://aclanthology.org/2026.findings-acl.141/)
- [Evaluation data: SEA-SafeguardBench](https://huggingface.co/datasets/aisingapore/SEASafeguardBench) (gated, access on request)
- [SEA-SafeguardBench paper](https://aclanthology.org/2026.findings-acl.194/)
