# Using Open Safety Models

A guide from the [ROOST Model Community](https://github.com/roostorg/model-community)

## Setting the Stage: Open AI Models in the Modern T&S Landscape

Modern trust & safety (T&S) teams require solutions to four functional archetypes, as outlined by the [DIRE framework](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5369158):

- **Detection:** Identifying potential risks in accounts, behaviors, and content through behavioral signals and existing databases (such as via hash matching, which identifies known threats based on their digital fingerprint). This is how you identify when something has gone wrong.

- **Investigation:** Analyzing broad attack patterns by evaluating context beyond individual entities, or diving deep into a single incident. This is how you understand why exactly things have gone wrong.

- **Review:** Assessing content against policies to determine appropriate next steps. This is how you determine the best course of action in nuanced cases.

- **Enforcement:** Taking actions and meeting reporting obligations. This is how you remove the threat from your platform.

AI models fine-tuned for safety can support teams across all four quadrants, but they're most well-suited to Detection and Review. Historically, detection tooling has been broadly commercially available. However, existing tools have failed on two major fronts. They are often (1) highly manual or reliant on user reporting and (2) fragmented or hard to integrate across internal tools and vendors.

We define open safety models as models that are free to access, available to deploy agnostic of platform, and fine-tuned specifically for a T&S use case; you can read more in the [ROOST Model Community (RMC) README](https://github.com/roostorg/model-community). These models offer a solution to our two main historic problems. For example: smaller developers may lack the resources necessary to run high-end private models; those who deal with sensitive personally identifiable information may want to keep their data private; and many teams may prefer to use their own policies or further fine-tune specific models on their policies.

These use cases have only expanded as AI capabilities have improved beyond traditional classifiers. But such advances have also rapidly introduced new threats to the ever evolving online safety landscape, ranging from novel AI-generated CSAM to agentic scams. Learning how to use safety models will be increasingly necessary to counter those threats. This guide aims to help new adopters start that learning process.

---

## Using Open Models in Your T&S Architecture

There are three particularly common use cases for modern open models in a Trust & Safety setting, although this is far from an exhaustive list:

1. **Content Detection:** You can use models to classify content according to an existing policy, which has historically been the typical use case for classifiers and often results in a binary decision (e.g., "policy violation" vs. "acceptable").
2. **Investigation & Review Support:** You can use models to support your investigation and review operations (whether automated or human-led), especially with reasoning models that can better handle abstracted or pattern-matched threats.
  - *Case Study:* Check out [Phoebe](https://github.com/haileyok/phoebe), an agentic solution that automates the entire investigation process by analyzing threat patterns and writing new rules to stop those threats via ROOST's Osprey engine.
3. **Iterative Improvement:** You can use models to iterate on your existing policies and processes, especially as you work to build policies that are increasingly interpretable by both human safety teams and AI.
  - *Case Study:* Participants at the Open Safeguard Hackathon built a [Policy Agent CLI](https://github.com/roostorg/model-community/tree/main/gpt-oss-safeguard/projects/open-safeguard-dec-2025/policy-agent-cli) to test, improve, and iterate on the performance of specific policies.

These three use cases are an oversimplification. They are also not mutually exclusive (and in fact often intersect). For example, platforms are increasingly considering a two-pronged approach to using safety models. The first model, usually smaller and lower latency, is a first layer of defense that renders judgments on easy to classify cases. Any edge cases that raise uncertainty can then be routed to the second model, often a higher performance model with reasoning capabilities, which proposes a recommended action for human review. The data from those decisions can (and should) then be used to iterate on policies.

Most importantly, these use cases are only the surface of open safety models' capabilities. Newer, more capable models unlock the possibility that T&S teams can automate more complex tasks, such as clustering content for deeper investigations, parsing multi-turn conversations for hidden threats (e.g., grooming), or the use of agentic moderators. These more advanced use cases are particularly valuable as your platform scales – they can help with account level evaluations and broader safety considerations (e.g., proactively stopping fraud campaigns).

---

## Choosing Your Model: Open Models vs. Closed Models

There is almost never a one-size-fits-all approach to tool selection, especially in a space this nuanced. You should choose the tool that best fits your needs, understanding the key trade-offs inherent in any model choice (performance, latency, price, etc.). Different models will have different strengths; some tasks may benefit from reasoning capabilities, while others may simply need the fastest and cheapest solution available.

One important consideration in model selection is whether you should be using an open model or a closed model. An oversimplified summary of the tradeoffs between those two is below:

**Open models often offer…**

- Highly customizable, fine-tuned solutions to policy or context specific questions
- Control over your data, content, and cost structure (particularly critical if handling sensitive information with compliance requirements)

**Closed models often offer…**

- Better coverage over a range of model sizes and capabilities (from large frontier models to smaller specialized models)
- Easier set-up given existing integrations (e.g., using Google's Content Safety API or OpenAI's Moderation API via [Coop](https://github.com/roostorg/coop))

This distinction does not cover the long list of specialized vendors who offer high quality, safety-oriented models (some of which were initially built upon open models). As mentioned above, multiple models can be used at different stages of your T&S stack; as such, a hybrid approach between closed and open models can be effective.

---

## Specific Tips & Tricks

### Getting Started: A Step-by-Step Guide to Running Open Models

- **Decide which model to use.** This should factor in questions such as the model's license (are there any restrictions that are prohibitive to you?), the model's size (will its latency be sufficient?), and the model's performance (often indicated via a model card or evaluation results).
- **Determine whether you want to run the model locally or via hosted inference API.** Local models afford full data privacy control and configurability, while APIs are often easier to set up.
  - This decision affects other considerations, such as latency – running the model locally on a Mac without a cloud GPU will likely lead to slower times than using an inference provider or setting up that GPU.
- **Test and refine the model before deploying.** Make sure that your model is performing sufficiently well on your policies before deploying it! To do this, you may want to:
  - Use or build a structured, high quality "golden" dataset to compare model performance against. This can often be a time and cost intensive practice, so don't be afraid to ask RMC peers if they have datasets to share!
  - Identify what metrics you are most worried about (e.g., precision vs. recall).
  - Understand whether errors stem from model performance or policy inconsistency.
  - Consider further model fine-tuning (if cost and time allow). Because out of the box classifiers are not customized to your specific policies, they may not always perform as you'd like without additional fine-tuning.
- For developers worried about managing costs, consider starting your T&S workflows with smaller, more lightweight models. Many basic, high volume tasks can be accomplished quickly and efficiently by smaller models. Consider where you really need to use a larger, more complex model. Validate your approach by running candidate models on sample batches of data before committing to a full deployment, and adjust your approach based on the results you find.

---

## Other Considerations

### A Note on Policies

Good policies are essential to effective deployment of an open safety model. Although there is not one answer to what a "good policy" looks like, having the wrong policy can lead to subpar results even if you have a perfect technical set up for using your models. Although ROOST does not write policies, some examples that have been open sourced through the RMC are below:

- [OpenAI's Teen Safety Policy Pack](https://github.com/roostorg/model-community/discussions/55)
- [Everyone.AI's Relational Positioning, Dependency & Exclusivity Policy](https://github.com/roostorg/model-community/discussions/57)

### Further Reading

This article is not a comprehensive summary. For other resources to help you get started, see the readings below:

- [Musubi: How to use LLMs for Content Moderation](https://www.musubilabs.ai/post/how-to-use-llms-for-content-moderation#real-world-examples-of-using-llms-for-content-moderation)
- [ROOST: Building Safety Infrastructure in the Open](https://roost.tools/blog/building-safety-infrastructure-in-the-open/)
- [Hugging Face: State of Open Source on Hugging Face: Spring 2026](https://huggingface.co/blog/huggingface/state-of-os-hf-spring-2026)

