# Open safety model landscape

An interactive version of the "Open safety models are specializing" graphic from the
[Shieldstral announcement](https://roost.tools/blog/mistral-shieldstral-joins-the-RMC/), drawn from a small
dataset in this folder so it can be kept up to date as new models are released.

- [`models.csv`](models.csv): one row per model; this is the only file you need to edit
- [`index.html`](index.html): a self-contained page (no build step or dependencies) that reads `models.csv` and draws the chart

The chart plots each model by **efficiency** (how much compute one decision takes) and **decision
transparency** (whether it returns only a decision or a reasoning trace with it). Colour shows the model type,
a dashed ring marks models that accept images, and bubble area tracks total parameter count. Hovering or
focusing a model shows its details; selecting it opens its model card.

## Viewing the chart

Browsers do not let a page read local files, so serve this folder over HTTP:

```sh
cd resources/open-safety-model-landscape
python3 -m http.server
```

Then open <http://localhost:8000/>. The folder also works as-is on any static host, such as GitHub Pages.

## Adding or updating a model

Add or edit a row in `models.csv`. The columns are:

| Column | Description |
| --- | --- |
| `model` | Display name |
| `developer` | Organization or authors that released the model |
| `year` | Release year |
| `approach` | One of `fixed-taxonomy`, `steerable-single-pass` or `steerable-reasoning` |
| `accepts_images` | `yes` or `no` |
| `params_b` | Total parameters, in billions |
| `active_params_b` | Active parameters per token, in billions, for mixture-of-experts models (otherwise empty) |
| `efficiency` | `0` (most efficient) to `1` (most compute-intensive) |
| `transparency` | `0` (decision only) to `1` (full reasoning trace) |
| `label_pos` | `above` or `below` the bubble, to avoid overlapping labels |
| `link` | Model card, or the paper if there is no public model card |
| `note` | Optional caveat shown in the tooltip and table |

`efficiency` and `transparency` are editorial placements that reflect each model's design, not measured
latency or benchmark scores. Cost per call is driven by how much a model generates per decision (a single
token, a short label, or a full reasoning trace) and by model size. Wrap a field in double quotes if it
contains a comma.
