# Agentic Safety Infrastructure

This directory contains reference templates for building trust & safety agents that can navigate an organization's data the way a human investigator would. The goal is to give safety teams a starting point they can fork, customize, and put in front of production, investigation, and assistant agents.

**These are templates, not contracts. Every safety team's data model is different. Every platform's entity hierarchy is different. The signals that actually drive detection live in different stores owned by different teams. Rather than force a common schema across organizations, the templates agree on the *shape* — how entities declare their edges, how signals declare their bindings, and how keys round-trip between the two — so an agent trained to read one fork of the template can read every fork.**

This work is part of the broader [ROOST](https://roost.tools) community effort to build open infrastructure for online safety. A longer write-up of the motivation lives in the [Agentic Safety blog post](https://roost.tools/blog). This README is the operational view.

## What's in here

```
agentic_infrastructure/
├── entity_schema/
│   ├── abstract_entity_schema.yaml   # Entities + edges (Actor, Content, Space, Cluster, Case, ...)
│   ├── signals.yaml                  # Signals that attach to entities (reputation, harm_labels, ...)
│   ├── discord_entity_schema.yaml    # Platform fork — Discord
│   └── youtube_entity_schema.yaml    # Platform fork — YouTube
├── harm_taxonomy.yaml                # Harm categories, severity, default actions, routing queues
└── playbooks/
    ├── investigation_coordinated_commenting.yaml   # Investigation persona — recommend only
    └── production_spam_autoaction.yaml             # Production persona — narrow auto-action
```

Entities have identity and a lifecycle. Signals are derived, mutable, and time-series. If you need to reference a thing by ID across time, it is an entity. If you compute it and overwrite it, it is a signal. The two files share one rule: every reference uses the same key shape, `{platform, platform_id}`.

An agent reading both files can traverse the graph (hop between entities) and enrich at each step (pull signals) without guessing at what exists or where to fetch it.

## Agent navigation paths

The four file types each answer one question. A playbook is the agent's only entry point; everything else is referenced.

| File | Question it answers |
|---|---|
| `playbooks/<name>.yaml` | What do I do, and in what order? |
| `entity_schema/abstract_entity_schema.yaml` | What entities and edges exist? |
| `entity_schema/signals.yaml` | What enrichments attach to each entity? |
| `entity_schema/<platform>_entity_schema.yaml` | How does my platform's data map onto the abstract entities? |
| `harm_taxonomy.yaml` | What category, severity, action, and queue should I recommend? |

**Investigation walk** (coordinated-commenting case):

```
playbooks/investigation_coordinated_commenting.yaml      # plan + thresholds
   │   trigger: Report
   ├──► entity_schema/abstract_entity_schema.yaml        # Report.target → Content
   │   walk Content.edges.author → Actor
   ├──► entity_schema/signals.yaml                       # Actor signals: reputation, cluster_memberships
   │   follow cluster_memberships.returns_entity_ref → Cluster
   ├──► entity_schema/abstract_entity_schema.yaml        # Space.edges.child_content → siblings
   ├──► entity_schema/signals.yaml                       # Content signals: similarity_hash, near_duplicate_cluster
   └──► harm_taxonomy.yaml                               # category, default_severity, default_actions, default_queue
        → open Case, route to needs_review
```

**Production walk** (high-confidence spam auto-action):

```
playbooks/production_spam_autoaction.yaml                # trigger + limits
   │   trigger: Content with classifier_scores.spam ≥ 0.95
   ├──► entity_schema/signals.yaml                       # classifier_scores, harm_labels, reputation, trust_tier
   ├──► entity_schema/abstract_entity_schema.yaml        # Content.author → Actor (is_verified, trust_tier)
   ├──► harm_taxonomy.yaml                               # forbidden_categories check + spam defaults
   └──► entity_schema/abstract_entity_schema.yaml        # write ModerationDecision + emit Event
```

The platform forks (`youtube_entity_schema.yaml`, `discord_entity_schema.yaml`) preserve the abstract vocabulary so the same playbook walk works against either without edits.

## The shape

Each entity declares:

- Its fields (typed)
- Its edges (named, with target entity and cardinality)
- The signals that attach to it (slot names resolved in `signals.yaml`)

Each signal declares:

- Which entity it attaches to
- Its type, cardinality, and freshness window
- Its source (the scorer, model, or pipeline)
- Its semantics — what the value actually means

A short walkthrough of the pattern in action — an investigation agent working a coordinated-commenting case — lives in [`entity_schema/BLOG_agent_schema.md`](entity_schema/BLOG_agent_schema.md).

## Forking for your platform

1. Copy `abstract_entity_schema.yaml` and `signals.yaml` into your org's repo.
2. Rename or remove entities you do not have. Add entities you do (e.g., `Transaction` for a payments platform).
3. Keep every entity's `edges:` and `signals:` blocks populated. Agents rely on them to plan walks.
4. Bind each signal slot to your actual scorer or store. The stub names common slots; the source field is yours to fill in.
5. Extend enums additively. Removing values is a breaking change for any agent that was relying on them.

You do not need to share your fork. The templates are open; your customizations can stay private.

## How to help

We are building this in the open because no single team has the full picture. The gaps are the interesting part.

- **Safety practitioners** — fork the templates and tell us where they break against your platform. Platform-specific concepts that do not map cleanly are exactly the signal we want.
- **Agent builders** — tell us which traversals show up often enough to be worth standardizing as playbook templates. Coordinated commenting is one; sextortion, card testing, and grooming patterns in gaming are three more that have come up in conversations.
- **Developers** — the schemas are the contract. Everything else — adapters, traversal engines, playbook runners, validators — is buildable on top. Contributions welcome.

Open an issue, submit a PR, or start a discussion if you want to help shape the direction. The adversaries are already navigating our infrastructure faster than we are; giving our agents the same map is how we catch up.
