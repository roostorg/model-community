# ROOST Model Community

The ROOST Model Community (RMC) brings together trust & safety practitioners with open safety model creators. We share evaluation outcomes, implementation tips, and other forms of feedback as we explore using and integrating these models into T&S workflows so that these models can be continuously improved and made more useful.

## Navigating this repo

This repository serves as the home of our GitHub Discussions and issues, plus a growing hub of resources. We're always interested in contributions, e.g. community-created guides for how to best use specific models or address specific use cases. If you'd like to contribute but don't know if it's something that fits, feel free to [open an issue](https://github.com/roostorg/model-community/issues)!

You can browse the current resources directly on GitHub:

- [gpt-oss-safeguard/](gpt-oss-safeguard): Resources and projects related to the open-weight safety model from OpenAI
- [projects/](projects): Interesting demos that are not RMC-model-specific
- [resources/](resources): Everything else!

## Join Us!

ROOST, our partners, and our community are hard at work to make an amazing space for collaboration, but we can’t do it alone!­ We need **you** to bring your ideas, expertise, and input to build something we can all be proud of.

- **[Start or join a discussion](https://github.com/roostorg/model-community/discussions) about model usage**. Review new evaluations, findings, and implementation tips, or share your own! Remember, if you're facing an issue, someone else probably is, too! This feedback is also critical for helping out both T&S practicioners as well as model creators who can use it to improve their models.

- **[Open an issue](https://github.com/roostorg/model-community/issues) to share how to improve this community**. For example, if you'd like to request a new discussion topic or issue label, or if you'd like to help add specific documentation or a guide added to the repo.

- **[Join our Discord](https://discord.gg/5Csqnw2FSQ) for real-time chat with other practicioners and model creators**. We're fairly active but you can engage exactly as much or little as works for you. It's a great less-formal and faster paced environment to share links, ask questions, etc.

- **[Join our office hours](https://github.com/roostorg/model-community/discussions/categories/office-hours) to connect directly with T&S peers and RMC model partners**. Get some face-to-face time, discuss what's working, share ideas, and more.
 
The RMC is open to all, but we're focused primarily on developers who are trying to increase the safety of their platforms, and are curious how AI models can help. These developers may be in a formal trust & safety role at a company, or may be an individual user running a self-hosted platform like a decentralized social network.

Developers who are focused on _creating_ open safety models are part of this ecosystem, but deeply engaging in that is a separate field in AI/ML engineering and not our primary purpose.

## What’s an “open safety model”?

There is no [single definition](https://arxiv.org/abs/2405.15802) of what makes a model “open.” Openness can apply to different parts of the stack—from datasets and weights to system-level safeguards and documentation—and in varying degrees. At ROOST, we believe that open source approaches can expand access to Trust & Safety tools and help create a safer internet. At the same time, the question of how open source licensing and norms apply to AI systems remains unsettled and continues to evolve.

For our purposes, we see “open safety models” as an AI model that:

- Has no monetary cost to access (e.g. downloadable weights are freely available)  
- Deployment of is platform agnostic   
- Is finetuned specifically for the purposes of Trust & Safety

Additionally, we prefer models that do not have non-commercial licenses and licenses that do not place restrictions on how outputs are handled. This is because many use cases for these models happen in commercial environments and have reporting requirements (such as CSAM). These license restrictions are incompatible with the Trust & Safety users we aim to support.

## RMC Partners

- OpenAI: [gpt-oss-safeguard](https://huggingface.co/collections/openai/gpt-oss-safeguard)

To discuss becoming an RMC Partner, email hello@roost.tools.

## Our approach to open model collaboration

The online safety field has long relied on machine learning to identify policy violations, but recent advances in safety-tuned AI models offer unprecedented capabilities for detecting and classifying harmful content based on customized rules. By making these powerful models openly accessible and integrating them into open source tooling, we aim to democratize AI-powered safety capabilities that were previously available only to well-resourced organizations.

Traditional open source development works well for software, but AI models present unique challenges; it requires sensitive training data, substantial computational resources, and has a fundamentally different development lifecycle than code.

Our community brings the open access and community spirit of open source software development to AI models, made possible by partnership with AI researchers and model creators. These partners commit to developing and openly releasing the weights of safety-finetuned models that are free to access, platform-agnostic in deployment, and have no commercial or output restrictions in their licenses. They also actively participate in the community, gathering feedback from practitioners and supporting their implementation journeys. In turn, ROOST cultivates a vibrant community-of-practice where safety teams share knowledge and strategies for successfully deploying these models in real-world scenarios.
