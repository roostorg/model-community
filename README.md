# ROOST Model Community

Welcome to the ROOST Model Community (RMC)! The RMC’s mission is to make open safety AI models accessible and beneficial to the safety community. To achieve that mission, we are building an ecosystem of developers, practitioners, model creators, and all who share the common goal of protecting online spaces. 

The RMC seeks to add tangible value for both active and potential users of open safety models and the developers of those models (called RMC Partners). Participants in the RMC benefit from:

- **A Community of Trust**: We only partner with high quality open safety models that you can trust; see our [eligibility criteria] for model selection. The RMC also stewards community conversations to encourage continued engagement and collective problem solving, via both routine events and casual conversations in our [Discord server].

- **Education and Implementation Support**: We share resources to help you better understand the open safety landscape, including introductory guides to AI for Trust & Safety and evaluation outcomes. Once you’re ready to use a specific model, the RMC also provides implementation tips and documentation, such as sample datasets or direct integrations with other ROOST projects.

- **Iterative Improvement**: By directly connecting users with model creators, the RMC helps model creators keep their models up to date with the latest feedback.

## Navigating this repo

This repository serves as the home to find resources for the community, including for RMC Partner models. If you'd like to contribute but don't know if it's something that fits, feel free to [open an issue]!

You can browse the current resources directly on GitHub:

- [gpt-oss-safeguard/](gpt-oss-safeguard): Resources and projects related to the open-weight safety model from OpenAI, our first RMC partner
- [projects/](projects): Interesting demos that are not RMC-model-specific
- [resources/](resources): Everything else

## Join us!

ROOST, our partners, and our community are hard at work to make an amazing space for collaboration, but we can’t do it alone!­ We need **you** to bring your ideas, expertise, and input to build something we can all be proud of.

- **Join our [bi-weekly office hours](https://github.com/roostorg/model-community/discussions/categories/office-hours) to connect directly with T&S peers and RMC model partners**. Get some face-to-face time, discuss what's working, share ideas, and more.

- **Join a project sprint**. We regularly host project sprints where community members work on a deliverable targeting a broader problem that the safety community faces. These are completely voluntary; contribute as much as you see fit.

- **[Start or join a discussion](https://github.com/roostorg/model-community/discussions) about model usage**. Review new evaluations, findings, model feedback, and implementation tips, or share your own! Remember, if you're facing an issue, someone else probably is, too! This feedback is also critical for helping out both T&S practitioners as well as model creators who can use it to improve their models.

- **[Open an issue] to share how to improve this community**. For example, if you'd like to request a new discussion topic or issue label, or if you'd like to help add specific documentation or a guide added to the repo.

- **Join our [Discord server] for real-time chat with other practitioners and model creators**. We're fairly active but you can engage exactly as much or little as works for you. It's a great less-formal and faster-paced environment to share links, ask questions, learn about upcoming events, and more.

The RMC is open to all, but we’re focused primarily on developers who are trying to increase the safety of their platforms, and are curious how AI models can help. These developers may be in a formal trust & safety role at a company, or may be an individual user running a self-hosted platform like a decentralized social network.

Developers who are focused on _creating_ open safety models are part of this ecosystem, but deeply engaging in that is a separate field in AI/ML engineering and not our primary purpose.

## What’s an “open safety model”?

There is no [single definition](https://arxiv.org/abs/2405.15802) of what makes a model “open.” Openness can apply to different parts of the stack—from datasets and weights to system-level safeguards and documentation—and in varying degrees. At ROOST, we believe that open source approaches can expand access to Trust & Safety tools and help create a safer internet. At the same time, the question of how open source licensing and norms apply to AI systems remains unsettled and continues to evolve.

For our purposes, we see “open safety models” as an AI model (including but not limited to both large language models and traditional classifiers) that:

- Has no monetary cost to access (e.g. downloadable weights are freely available)  
- Deployment of is platform agnostic   
- Is finetuned specifically for the purposes of Trust & Safety

Additionally, we prefer models that do not have non-commercial licenses and licenses that do not place restrictions on how outputs are handled. This is because many use cases for these models happen in commercial environments and have reporting requirements (such as CSAM). These license restrictions are incompatible with the Trust & Safety users we aim to support.

Although there are many open safety models, we hold a specific bar for RMC Partners. To formally become an RMC partner, a model must meet our [eligibility criteria]. We work closely with RMC partners to ensure that their models meet these criteria before adding them to the community. For any questions about these criteria, please feel free to reach out – we designed these criteria to meet the community’s needs and are open to feedback!

## RMC Partners

- OpenAI: [gpt-oss-safeguard](https://huggingface.co/collections/openai/gpt-oss-safeguard)

RMC Partners receive a variety of benefits, including:

- Promotion of their model (e.g., via demos during Office Hours, support on events)
- Contributions to model onboarding resources (e.g., documentation, implementation tips, integrations with other ROOST tools as desired)
- Support in surfacing product feedback

We seek to build a friendly, inclusive community for all model developers. Non-partner model creators are always encouraged to participate in RMC programming, even if they do not strictly meet the partner eligibility criteria.

To discuss becoming an RMC Partner, email hello@roost.tools.

## Our approach to open model collaboration

The online safety field has long relied on machine learning to identify policy violations, but recent advances in safety-tuned AI models offer unprecedented capabilities for detecting and classifying harmful content based on customized rules. By making these powerful models openly accessible and integrating them into open source tooling, we aim to democratize AI-powered safety capabilities that were previously available only to well-resourced organizations.

Traditional open source development works well for software, but AI models present unique challenges; it requires sensitive training data, substantial computational resources, and has a fundamentally different development lifecycle than code.

Our community brings the open access and community spirit of open source software development to AI models, made possible by partnership with AI researchers and model creators. These partners commit to developing and openly releasing the weights of safety-finetuned models that are free to access, platform-agnostic in deployment, and have no commercial or output restrictions in their licenses. They also actively participate in the community, gathering feedback from practitioners and supporting their implementation journeys. In turn, ROOST cultivates a vibrant community-of-practice where safety teams share knowledge and strategies for successfully deploying these models in real-world scenarios.

[discord server]: https://discord.gg/UXmBqy7kFX
[eligibility criteria]: https://docs.google.com/spreadsheets/d/1gkRwjCYFlYrah1WBZu96iWzAVUj555WO4rxR5NswzEI/edit?gid=667042538#gid=667042538
[open an issue]: https://github.com/roostorg/model-community/issues
