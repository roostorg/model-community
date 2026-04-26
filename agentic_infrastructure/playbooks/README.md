# Playbooks

Per-harm investigation and production playbooks. Each playbook is a
declarative recipe an agent reads before it starts work. Playbooks
reference entities and edges from `../entity_schema/abstract_entity_schema.yaml`,
signal slots from `../entity_schema/signals.yaml`, and category defaults from
`../harm_taxonomy.yaml`.

Two personas, two scopes:

| File | Persona | Authority |
|---|---|---|
| `investigation_coordinated_commenting.yaml` | Investigation agent | Recommends only. Opens a Case, routes to a human queue. |
| `production_spam_autoaction.yaml`            | Production agent     | Acts within tight thresholds. Logs every action to Coop. |

## Query catalog convention

Per the blog's "pre-approved query catalog" principle, agents do not
write SQL. Each playbook declares the queries it is allowed to call by
name; the executor owns the actual SQL. The `queries:` block in each
playbook holds **placeholder SQL** that an integrator replaces with
their warehouse's real schema. Parameter shape and return columns
should be preserved so the playbook's `walk:` steps still bind.

## Adding a playbook

1. Copy one of the two examples.
2. Name the file `<persona>_<harm_id>.yaml` where `harm_id` is a
   category from `harm_taxonomy.yaml`.
3. Keep `walk:` steps small and ordered. Each step names the entity,
   the query, and what it produces.
4. Bind every `decide:` rule to a category default from the taxonomy
   unless you are intentionally overriding policy — and say why.
