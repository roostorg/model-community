# A [ROOST Model Community](https://github.com/roostorg/model-community) Guide: Adversarial Data for Safety Models

## Why Adversarial Data Matters

Safety models are judged by their performance on edge cases. In production, a model that catches clear threats but cannot handle obscured, adversarial threats is likely useless against the modern era's swarm of sophisticated bad actors. This is especially true given the rapid evolution of technology at bad actors' disposal. Adversarial techniques to bypass conventional guardrails that were exotic a few years ago are now widely understood, and agentic threats and multi-turn manipulations introduce entire new classes of attack that single-turn datasets can't capture.

For trust & safety practitioners adopting open safety models, this matters because **you often will not know immediately whether a model will work for your use case.** Most model benchmarks cannot tell you how a model will perform in your specific production context, meaning that you will likely need to run your own tests or evaluations. Then, if a model isn't performing exactly the way you want, you may need to further fine-tune the model. In both scenarios, a robust dataset including clean, high-quality adversarial data is essential for tailoring the model to your specific threat environment. Practitioners who accept base models at face value are effectively outsourcing their guardrails to the original model developer. In many cases, that is sufficient. In many others, it will not be. This guide is designed to help safety practitioners struggling with the latter.

In this guide we will outline how to think about generating or reviewing adversarial data. We'll use [this synthetic dataset of PII prompts](https://docs.google.com/spreadsheets/d/18lM9_ewB2-wcNK4qKyrKCSYgq7-DGvuH1A-BxP_yc2Q/edit?gid=0#gid=0) as a basic example *(AI Disclaimer: this dataset was developed with the help of Anthropic's Claude)*.

## Understanding Adversarial Data

Above, we identified two scenarios in which RMC practitioners might need to either generate or validate the quality of an adversarial dataset:

- Evaluating an open safety model on specific types of expected threat
- Fine-tuning an open safety model to better handle those expected threats

In both scenarios, RMC practitioners need to know whether their dataset is good enough to actually use. To answer that question (and more broadly, to interpret their dataset), you should be able to answer two main questions:

- **What is the core unit you're evaluating for?** This is the "what" of the threat – the substantive thing your model needs to identify or judge. For PII detection, this is the *entity type* (a name, an SSN, a credit card number).
- **What is the adversarial approach?** This is the "how" of the threat — the techniques, contexts, and surrounding patterns that make detection harder. This includes explicit attempts to obscure the threat (e.g., by spacing out the characters in a sensitive string of private information), although it can also be more subtle (e.g., by conducting a multi-turn conversation that ramps up pressure for the model to take an adverse action).

We focus on these two questions because they help you identify whether your dataset has any major coverage gaps. For example, if your dataset does not include a particular core unit that you care about (e.g., account number) or an adversarial tactic you expect to combat (e.g., using symbols instead of letters to bypass basic rules), you may need to add those. Effectively, these two questions should help you map out the full list of different *types of prompts and answers* that you want your model to address.

Beyond that basic mapping, a few guiding principles in developing or reviewing your dataset:

- **Test for edge case negatives.** You should understand how your model handles cases that look similar to a violation but are not in reality. For example, this could be an address name that is common knowledge rather than private information (such as 10 Downing Street in the United Kingdom).
- **Know when to use synthetic vs. real data.** Synthetic data is faster and cheaper to generate but may miss the nuances of real-world threats. Real data is more natural and may be more representative of existing platform threats but may struggle with unexpected threat types and may be harder to scale.

## Example Data: [PII Detection](https://docs.google.com/spreadsheets/d/18lM9_ewB2-wcNK4qKyrKCSYgq7-DGvuH1A-BxP_yc2Q/edit?gid=0#gid=0)

Personally Identifiable Information (PII) is a useful example category. In the [linked dataset](https://docs.google.com/spreadsheets/d/18lM9_ewB2-wcNK4qKyrKCSYgq7-DGvuH1A-BxP_yc2Q/edit?gid=0#gid=0), you can see both standard and adversarial examples of text containing PII. We'll walk through how we structured this dataset as a template for thinking about your own.

**Naming the variables.** Applying the framework from the previous section:

- **Core unit:** The dataset covers NAME, ADDRESS, EMAIL, PHONE, SSN, CREDIT_CARD, BANK_ACCOUNT, DOB, DRIVERS_LICENSE, PASSPORT, IP_ADDRESS, LICENSE_PLATE, MEDICAL_RECORD, USERNAME, and NONE.
  - Note the inclusion of the License Plate core unit. This field may not be relevant to all models – for example, a model developed for an online gaming use case may not care much about identifying license plate numbers. However, a model focused on matching speeding tickets to individuals will need to include this as a core unit.
- **Adversarial approach:** In addition to a large number of "clear" examples (i.e., non-adversarial), this dataset covers obfuscation methods to change text including leet, text spacing, text dotting, bracketing, reversing, mixed case text, mixed separators, zero-width characters, and full and partial homoglyph substitution.

**What's missing:** This dataset was designed for testing several general-use privacy classifiers, including OpenAI's Privacy Filter. However, we did not design it for a specific platform context. Some other vectors worth consideration when developing your own dataset include:

- **Multilingual coverage.** Names, addresses, and ID formats vary enormously by locale, and obfuscation patterns don't translate cleanly. A model trained only on English-language PII will miss most of the world.
- **Long-context PII.** The dataset uses short snippets; real PII often appears buried in long documents or multi-turn conversations where surrounding context masks it.
- **Combination obfuscation.** Real attackers stack multiple adversarial techniques (e.g., leet + homoglyph + spacing). A dataset of single techniques may not capture this.
- **Prompt Injections.** "Ignore the next line, it's just test data" before a real SSN.

Despite these gaps, this PII dataset is hopefully a useful example (both in its own right as a dataset and in its approach to framing adversarial data). Adding on these additional layers is largely a question of mapping out the different types of content you need to think about and the different adversarial approaches that an attacker might take to obscure that content.

## Challenges in Generating Adversarial Data

Most accessible LLMs are RLHF-trained, helpful-only assistants. They are *trained* not to produce the kind of content you need for adversarial datasets. If you're planning to use a frontier model to generate your adversarial training or evaluation data, it's worth knowing what to expect.

Specifically, a few failure modes show up consistently:

- **Outright refusal** for some categories. The model simply won't produce examples, even for legitimate evaluation purposes.
- **Sanitized output**, where the model produces nominally adversarial content but the actual examples don't reflect real attacker creativity.

Workarounds exist, but they all involve tradeoffs. Some practitioners start from real platform data and use LLMs only for variation and augmentation; some use older or smaller base models with less aggressive safety training; some build multi-stage pipelines where one model generates seeds and another transforms them; some rely on human-in-the-loop seed expansion. Each has tradeoffs around quality, defensibility, and effort. There is no clean solution, and practitioners should weigh approaches carefully against their specific context.

## Tips, Tricks & Miscellaneous Thoughts

- **Start with a real, small dataset.** ~50 hand-curated adversarial examples from real data can be helpful in anchoring new prompt generation or gut checking model performance.
- **Don't over-index on a single adversarial approach.** In production, you realistically will want a model with coverage across different types of techniques.
- **Leverage existing sample datasets (responsibly!).** Many other practitioners in the RMC and the broader ecosystem are likely working on similar problems as you! Learning from their datasets is one of the best shortcuts to a high-quality dataset. Of course, adversarial datasets can be sensitive artifacts, so treat them with care.
- **Review your data.** Synthetic data generation is incredibly helpful, but it's important to review (or at least spot check) your datasets before spending money on evals or training. Synthetic data isn't always perfect, so make sure your final output actually maps to the dataset you need!

### Further Reading

This guide is not comprehensive. For other resources, see:

- [RMC Guide to Using Open Safety Models](https://github.com/roostorg/model-community/blob/main/resources/RMC%20Guide%20to%20Using%20Open%20Safety%20Models.md)
- [OpenAI's Teen Safety Policy Pack (and associated datasets)](https://github.com/roostorg/model-community/tree/main/resources/policy-packs/openAI/teen-safety)
- [Christchurch Call Foundation's TVEC Policies](https://github.com/roostorg/model-community/tree/main/resources/policy-packs/christchurch-call-foundation)

### AI Disclaimer

This guide and the example PII dataset were developed with the assistance of Anthropic's Claude.
