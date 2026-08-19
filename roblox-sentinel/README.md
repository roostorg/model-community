# Roblox Sentinel

- [Roblox Sentinel on GitHub](https://github.com/Roblox/Sentinel)
- [Roblox Announcement](https://about.roblox.com/newsroom/2025/08/open-sourcing-roblox-sentinel-preemptive-risk-detection)

Sentinel is a Python library from the Roblox Safety Toolkit for detecting extremely rare classes of text. It scores each message by embedding similarity against contrastive positive and negative examples, then aggregates those scores across one source's recent messages, surfacing patterns that no single message would reveal.

Development happens in [Roblox/Sentinel](https://github.com/Roblox/Sentinel); please file issues and pull requests there.

### Helpful Resources
- [Installation and quick start](https://github.com/Roblox/Sentinel#installation)
- [How it works](https://github.com/Roblox/Sentinel#how-it-works)
- [Worked example notebook: detecting hate speech](https://github.com/Roblox/Sentinel/blob/main/examples/sentinel_against_hate.ipynb)
- [Choosing an aggregation strategy](https://github.com/Roblox/Sentinel#choosing-an-aggregation-strategy)
- [Tuning against your own data](https://github.com/Roblox/Sentinel#simulation-based-tuning)
