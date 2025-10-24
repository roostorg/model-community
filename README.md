# ROOST Model Community
The ROOST Model Community (RMC) brings together Trust & Safety practitioners using open source AI models to protect online spaces. We share evaluation outcomes and implementation tips as we explore integrating open safety models into T&S workflows. With the support of open safety model creators, we also help them improve and refine their models, similar to the user feedback cycle of open source software. 

**Join Us:**

* **Shape the next generation of open safety models via direct feedback**  
  * [RMC Partners](#rmc-partners) are open safety model creators who are eager to iterate on their models with the RMC.   
    * To share model feedback, please open an [*Issue*](https://github.com/roostorg/open-models/issues) and use the corresponding model Label.
* **Put on your explorer hat and venture into the field of open source AI for safety**  
  * Review new evaluations, findings, and implementation tips, or share your own\!   
    * You can start a new conversation or dive into existing ones on our [*Discussions page*](https://github.com/roostorg/open-models/discussions)  
    * Discussions are organized by Evaluation, Implementation, and general discussion about the RMC.   
      * When adding a new discussion to Evaluations, please Label it with the model  
      * Implementation is organized by model  
      * (Open an Issue to request new categories or Labels\!)   
  * Build community-created guides for how to best use specific models or address specific use cases.   
    * Aggregate tips and information in [*Discussion*](https://github.com/roostorg/open-models/discussions) threads, then suggest a new guide via [*Issues*](https://github.com/roostorg/open-models/issues).   
    * Or, skip to the fun part and start a new guide by submitting a [*Pull Request*](https://github.com/roostorg/open-models/pulls).  
* **Improve your safety experience by learning from and with peers tackling similar problems**  
  * Join our bi-weekly office hours\! Share what’s working and what’s not in real time, and connect directly with T\&S peers and RMC model partners. Learn more via the [*ROOST \#safety-models Discord channel*](https://discord.gg/5Csqnw2FSQ).  
* **Craft the RMC experience**
  * Help us build a community for using open AI models in Trust & Safety\! If you have ideas of how we can improve the RMC, open an [*Issue*](https://github.com/roostorg/open-models/issues) with the “RMC” Label.
* **Continue the discussion on the [*\#safety-models ROOST Discord channel*](https://discord.gg/5Csqnw2FSQ)\!**
  * Here, you can ask questions, troubleshoot challenges, celebrate your wins, and find fellow safety enthusiasts.
 
RMC is open to all, but its purpose and deliverables are focused on developers who are trying to increase the safety of their platform, and are curious how AI models can help. These developers may be in a formal Trust & Safety role at a company, or may be an individual user running a self-hosted platform, like a decentralized social network. 

Developers who are focused on *creating* open safety models are part of this ecosystem, but deeply engaging in that is a separate field in AI/ML engineering and not RMC’s primary purpose.

## What’s an “open safety model”?

There is no [single definition](https://arxiv.org/abs/2405.15802) of what makes a model “open.” Openness can apply to different parts of the stack—from datasets and weights to system-level safeguards and documentation—and in varying degrees. At ROOST, we believe that open source approaches can expand access to Trust & Safety tools and help create a safer internet. At the same time, the question of how open source licensing and norms apply to AI systems remains unsettled and continues to evolve.

For RMC’s purposes, we see “open safety models” as an AI model that:

* Has no monetary cost to access (eg downloadable weights are freely available)  
* Deployment of is platform agnostic   
* Is finetuned specifically for the purposes of Trust & Safety

Additionally, we prefer models that do not have non-commercial licenses and licenses that do not place restrictions on how outputs are handled. This is because many use cases for these models happen in commercial environments and have reporting requirements (such as CSAM). These license restrictions are incompatible with the Trust & Safety users RMC aims to support. 

## RMC Partners
* OpenAI: [gpt-oss-safeguard](https://huggingface.co/collections/openai/gpt-oss-safeguard)

To discuss being an RMC Partner, email hello@roost.tools.

## RMC’s approach to open model collaboration

The online safety field has long relied on machine learning to identify policy violations, but recent advances in safety-tuned AI models offer unprecedented capabilities for detecting and classifying harmful content based on customized rules. By making these powerful models openly accessible and integrating them into open source tooling, RMC aims to democratize AI-powered safety capabilities that were previously available only to well-resourced organizations.

Traditional open source development works well for software, but AI models present unique challenges; it requires sensitive training data, substantial computational resources, and has a fundamentally different development lifecycle than code.

RMC brings the open access and community spirit of open source software development to AI models, made possible by partnership with AI researchers and model creators. These partners commit to developing and openly releasing the weights of safety-finetuned models that are free to access, platform-agnostic in deployment, and have no commercial or output restrictions in their licenses. They also actively participate in the community, gathering feedback from practitioners and supporting their implementation journeys. In turn, ROOST cultivates a vibrant community-of-practice where safety teams share knowledge and strategies for successfully deploying these models in real-world scenarios.
