# Copyright 2025 Roblox Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simulation and tuning harness for Sentinel.

This module helps you answer a practical question: *which aggregation
("summarize metric") and which hyperparameters work best for my data?*

It mirrors the workflow used to evaluate Sentinel in production, but with none
of the production infrastructure (no Ray, S3, Hydra, or experiment trackers) so
it stays easy to read and depends only on numpy.

There are two very different "metrics" involved, and it helps to keep them
separate:

1. The **summarize metric** (a.k.a. the aggregation function): how a group's
   many per-observation scores are collapsed into a single affinity number.
   Sentinel ships several of these in :mod:`sentinel.score_formulae`
   (``skewness``, ``mean_of_positives``, ``top_k_mean``, ``percentile_score``,
   ``softmax_weighted_mean``, ``max_score``). This is *not* a ranking.

2. The **evaluation metric**: once every group has one affinity number and a
   known label, how good is that separation? Ranking (where do the known
   positives land in the leaderboard?) is only one option. This harness reports
   three families so you can tune for whatever you care about:

   - ranking (ROC-AUC, recall@N / precision@N, average-rank ratio),
   - threshold / classification (precision, recall, F1 at a cutoff),
   - separation / distribution (mean separation, Cohen's d, KS statistic).

Typical usage::

    from sentinel.simulation import LabeledGroup, score_groups, compare_aggregators

    groups = [
        LabeledGroup(name="episode_a", label=1, observations=[...segments...]),
        LabeledGroup(name="episode_b", label=0, observations=[...segments...]),
        ...
    ]

    # Expensive step (runs the sentence model) -- done once.
    scored = score_groups(index, groups, top_k=5)

    # Cheap step -- compare every aggregator without re-encoding.
    rows = compare_aggregators(scored)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from sentinel.score_formulae import (
    max_score,
    mean_of_positives,
    percentile_score,
    skewness,
    softmax_weighted_mean,
    top_k_mean,
)

if TYPE_CHECKING:
    # Imported only for type hints. Kept out of runtime imports so this module
    # stays lightweight (numpy-only) and does not pull in torch / transformers.
    from sentinel.sentinel_local_index import SentinelLocalIndex

LOG = logging.getLogger(__name__)


# A convenient name -> function map of the built-in summarize metrics, so a
# caller (or a notebook) can iterate over "all aggregators" in one line.
DEFAULT_AGGREGATORS: Dict[str, Callable[[np.ndarray], float]] = {
    "skewness": skewness,
    "mean_of_positives": mean_of_positives,
    "top_k_mean": top_k_mean,
    "percentile_score": percentile_score,
    "softmax_weighted_mean": softmax_weighted_mean,
    "max_score": max_score,
}


@dataclass
class LabeledGroup:
    """A collection of observations from a single source, with a known label.

    Think of one group as, for example, one podcast episode (its observations
    are the episode's text segments) or one user (their recent messages).

    Attributes:
        name: A human-readable identifier for the group.
        label: ``1`` if this group belongs to the rare/positive class (e.g. a
            hateful episode), ``0`` if it belongs to the common/negative class.
        observations: The individual texts that make up this group.
    """

    name: str
    label: int
    observations: Sequence[str]


@dataclass
class GroupObservationScores:
    """Raw per-observation scores for one group (computed once, reused often).

    Attributes:
        name: The group's identifier (copied from :class:`LabeledGroup`).
        label: The group's label (copied from :class:`LabeledGroup`).
        observation_scores: 1-D array of per-observation scores, *before* any
            ``min_score_to_consider`` threshold is applied. Keeping them raw lets
            us sweep thresholds later without re-running the sentence model.
    """

    name: str
    label: int
    observation_scores: np.ndarray


# Columns run_grid_search attaches on top of whatever evaluate_groups returns. Every
# index-describing column is prefixed, because evaluate_groups already returns an
# "n_positive" meaning the number of positive *groups* in the evaluation set; an
# unprefixed index size once overwrote it and silently turned evaluation metadata into
# an index size. _add_columns turns any repeat of that mistake into an error.
COLUMN_TOP_K = "top_k"
COLUMN_INDEX_N_POSITIVE = "index_n_positive"
COLUMN_INDEX_NEG_TO_POS_RATIO = "index_neg_to_pos_ratio"
COLUMN_INDEX_N_POSITIVE_ACTUAL = "index_n_positive_actual"
COLUMN_INDEX_N_NEGATIVE_ACTUAL = "index_n_negative_actual"

GRID_SEARCH_COLUMNS = (
    COLUMN_TOP_K,
    COLUMN_INDEX_N_POSITIVE,
    COLUMN_INDEX_NEG_TO_POS_RATIO,
    COLUMN_INDEX_N_POSITIVE_ACTUAL,
    COLUMN_INDEX_N_NEGATIVE_ACTUAL,
)


def _add_columns(row: Dict[str, Any], **columns: Any) -> Dict[str, Any]:
    """Add columns to a result row, refusing to overwrite one that already exists.

    Rows are plain dicts so they drop straight into ``pandas.DataFrame``, but that also
    means a second write to the same key destroys the first with no error. That is how
    an index size once replaced the positive-group count. Raising here turns a silent
    data loss into a failing test.

    Args:
        row: The row to extend, modified in place.
        **columns: Column name to value.

    Returns:
        The same row, for convenient chaining.

    Raises:
        ValueError: If any column is already present in the row.
    """
    clashes = sorted(name for name in columns if name in row)
    if clashes:
        raise ValueError(
            f"Refusing to overwrite existing result column(s): {', '.join(clashes)}. "
            f"Two different measurements are competing for one name; rename the new "
            f"one rather than letting it replace the old value."
        )
    row.update(columns)
    return row


def encode_observations(
    index: "SentinelLocalIndex",
    groups: Sequence[LabeledGroup],
    *,
    show_progress_bar: bool = False,
    encoding_additional_kwargs: Optional[Mapping[str, object]] = None,
) -> Dict[str, np.ndarray]:
    """Embed every group's observations once, so many scoring passes can share them.

    Encoding is the only expensive step in a simulation, and an observation's embedding
    depends on the encoder, never on the index it is scored against. A sweep that varies
    index size, ratio or ``top_k`` therefore re-computes identical numbers on every pass
    unless the embeddings are hoisted out, which is what this does.

    The result stays valid across every index produced by
    :meth:`~sentinel.sentinel_local_index.SentinelLocalIndex.subsample`, because a
    subsampled copy shares the parent's sentence model and encoding kwargs.

    Args:
        index: A loaded :class:`~sentinel.sentinel_local_index.SentinelLocalIndex`.
        groups: The labeled groups whose observations should be embedded.
        show_progress_bar: Whether to show the encoder progress bar.
        encoding_additional_kwargs: Extra keyword arguments forwarded to the encoder.

    Returns:
        Group name to the embeddings of that group's observations, in observation order.
        Groups with no observations are omitted. Empty if ``index`` cannot encode on its
        own, in which case callers simply fall back to encoding per pass.
    """
    if encoding_additional_kwargs is None:
        encoding_additional_kwargs = {}

    sentence_model = getattr(index, "sentence_model", None)
    if sentence_model is None or not hasattr(index, "encoding_kwargs"):
        # This harness deliberately accepts any index-like object, including test
        # doubles that only implement calculate_rare_class_affinity. Those cannot be
        # pre-encoded, so skip the optimisation rather than refusing to run.
        LOG.debug(
            "Index exposes no sentence model, so observations cannot be pre-encoded; "
            "scoring will encode per pass."
        )
        return {}

    encoding_kwargs = dict(index.encoding_kwargs)
    encoding_kwargs["show_progress_bar"] = show_progress_bar
    encoding_kwargs.update(encoding_additional_kwargs)

    embeddings: Dict[str, np.ndarray] = {}
    for group in groups:
        observations = list(group.observations)
        if not observations:
            continue
        embeddings[group.name] = sentence_model.encode(
            observations,
            **encoding_kwargs,
        )

    LOG.info("Encoded observations for %d groups", len(embeddings))
    return embeddings


def score_groups(
    index: "SentinelLocalIndex",
    groups: Sequence[LabeledGroup],
    *,
    top_k: int = 5,
    observation_embeddings: Optional[Mapping[str, np.ndarray]] = None,
    show_progress_bar: bool = False,
    encoding_additional_kwargs: Optional[Mapping[str, object]] = None,
) -> List[GroupObservationScores]:
    """Score every observation in every group, once.

    This is the expensive part of a simulation because it runs the sentence
    model to embed and score each observation. Aggregation and evaluation happen
    separately and cheaply, so you can compare many summarize metrics and
    thresholds afterward without paying the encoding cost again.

    Scores are computed with ``min_score_to_consider=0.0`` so the raw values are
    preserved; :func:`evaluate_groups` applies any threshold later.

    Args:
        index: A loaded :class:`~sentinel.sentinel_local_index.SentinelLocalIndex`.
        groups: The labeled groups to score.
        top_k: Number of nearest neighbors used per observation. Changing this
            changes the per-observation scores, so it requires re-scoring.
        observation_embeddings: Optional embeddings from
            :func:`encode_observations`, keyed by group name. Supplying them skips the
            encoding step, which is what makes repeated scoring passes cheap. A group
            missing from the mapping is encoded normally.
        show_progress_bar: Whether to show the encoder progress bar.
        encoding_additional_kwargs: Extra keyword arguments forwarded to the
            encoder.

    Returns:
        One :class:`GroupObservationScores` per input group, in the same order.
    """
    if encoding_additional_kwargs is None:
        encoding_additional_kwargs = {}

    scored: List[GroupObservationScores] = []
    for group in groups:
        observations = list(group.observations)

        if len(observations) == 0:
            scored.append(
                GroupObservationScores(
                    name=group.name,
                    label=int(group.label),
                    observation_scores=np.array([], dtype=float),
                )
            )
            continue

        # We only need the per-observation scores here, so we disable the
        # explainability extras for speed and pass a trivial aggregation
        # function (its output is ignored).
        # Only forwarded when there is something to forward, so an index-like object
        # that does not know about sample_embeddings keeps working unchanged.
        cached = (
            None
            if observation_embeddings is None
            else observation_embeddings.get(group.name)
        )
        cached_kwargs = {} if cached is None else {"sample_embeddings": cached}

        result = index.calculate_rare_class_affinity(
            observations,
            top_k=top_k,
            min_score_to_consider=0.0,
            aggregation_function=max_score,
            show_progress_bar=show_progress_bar,
            explain=False,
            include_neighbors=False,
            encoding_additional_kwargs=encoding_additional_kwargs,
            **cached_kwargs,
        )

        observation_scores = np.asarray(
            list(result.observation_scores.values()), dtype=float
        )
        scored.append(
            GroupObservationScores(
                name=group.name,
                label=int(group.label),
                observation_scores=observation_scores,
            )
        )

    return scored


def _apply_threshold(scores: np.ndarray, min_score_to_consider: float) -> np.ndarray:
    """Zero out per-observation scores below the threshold (cheap, no encoding).

    This reproduces exactly what ``calculate_rare_class_affinity`` does with its
    ``min_score_to_consider`` argument, but applied to already-computed scores.
    """
    if scores.size == 0 or min_score_to_consider <= 0.0:
        return scores
    return np.where(scores < min_score_to_consider, 0.0, scores)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks of ``values`` (smallest = rank 1), averaging ties.

    Ties receive the average of the ranks they span (the standard convention
    used by the Mann-Whitney statistic), which keeps metrics stable when many
    groups share the same score (e.g. lots of zeros).
    """
    n = values.size
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        # Average 1-based rank for the tied block [i, j].
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def _roc_auc(affinities: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank-based (Mann-Whitney) formula. numpy-only.

    Equals the probability that a randomly chosen positive group scores higher
    than a randomly chosen negative one. Ties count as 0.5. Returns ``nan`` if
    either class is missing.
    """
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(affinities)
    sum_ranks_pos = float(np.sum(ranks[labels == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _ranking_metrics(
    affinities: np.ndarray, labels: np.ndarray, top_n: Optional[int]
) -> Dict[str, float]:
    """Ranking family: leaderboard-style metrics (production's original choice)."""
    n_groups = int(labels.size)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    metrics: Dict[str, float] = {"roc_auc": _roc_auc(affinities, labels)}

    # Leaderboard rank where 1 = highest affinity. Rank the *negated* affinities
    # so the largest affinity gets the smallest rank number. Ties are averaged.
    leaderboard_rank = _average_ranks(-affinities)

    # recall@N / precision@N over the top-N of the leaderboard. Default N is the
    # number of positives (a.k.a. R-precision), which mirrors production intent.
    top_n_eff = n_pos if top_n is None else int(top_n)
    top_n_eff = max(0, min(top_n_eff, n_groups))
    metrics["top_n"] = float(top_n_eff)
    if top_n_eff > 0 and n_pos > 0:
        in_top = labels[leaderboard_rank <= top_n_eff]
        tp_in_top = int(np.sum(in_top == 1))
        metrics["precision_at_n"] = tp_in_top / top_n_eff
        metrics["recall_at_n"] = tp_in_top / n_pos
    else:
        metrics["precision_at_n"] = float("nan")
        metrics["recall_at_n"] = float("nan")

    metrics["avg_rank_positive"] = (
        float(np.mean(leaderboard_rank[labels == 1])) if n_pos > 0 else float("nan")
    )
    metrics["avg_rank_negative"] = (
        float(np.mean(leaderboard_rank[labels == 0])) if n_neg > 0 else float("nan")
    )
    # Lower is better: positives should have smaller (better) ranks than negatives.
    if n_pos > 0 and n_neg > 0 and metrics["avg_rank_negative"] > 0:
        metrics["rank_ratio"] = (
            metrics["avg_rank_positive"] / metrics["avg_rank_negative"]
        )
    else:
        metrics["rank_ratio"] = float("nan")

    return metrics


def _classification_at_threshold(
    affinities: np.ndarray, labels: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Confusion-matrix counts and precision/recall/F1 at a fixed cutoff."""
    predicted_positive = affinities >= threshold
    tp = int(np.sum(predicted_positive & (labels == 1)))
    fp = int(np.sum(predicted_positive & (labels == 0)))
    fn = int(np.sum(~predicted_positive & (labels == 1)))
    tn = int(np.sum(~predicted_positive & (labels == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


def _threshold_metrics(
    affinities: np.ndarray, labels: np.ndarray, decision_threshold: Optional[float]
) -> Dict[str, float]:
    """Threshold / classification family.

    If ``decision_threshold`` is given, metrics are reported at that exact
    cutoff. If it is ``None``, we sweep the candidate cutoffs (the sorted unique
    affinities) and report the one that maximizes F1 -- a useful "best
    achievable" signal when tuning.
    """
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return {
            "decision_threshold": float("nan"),
            "tp": float("nan"),
            "fp": float("nan"),
            "fn": float("nan"),
            "tn": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "false_positive_rate": float("nan"),
        }

    if decision_threshold is not None:
        metrics = _classification_at_threshold(
            affinities, labels, float(decision_threshold)
        )
        metrics["decision_threshold"] = float(decision_threshold)
        return metrics

    best_metrics: Optional[Dict[str, float]] = None
    best_threshold = float("nan")
    for candidate in np.unique(affinities):
        candidate_metrics = _classification_at_threshold(affinities, labels, candidate)
        if best_metrics is None or candidate_metrics["f1"] > best_metrics["f1"]:
            best_metrics = candidate_metrics
            best_threshold = float(candidate)

    assert best_metrics is not None  # np.unique is non-empty when groups exist
    best_metrics["decision_threshold"] = best_threshold
    return best_metrics


def _ks_statistic(positives: np.ndarray, negatives: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max gap between the two CDFs."""
    grid = np.sort(np.concatenate([positives, negatives]))
    cdf_pos = np.searchsorted(np.sort(positives), grid, side="right") / positives.size
    cdf_neg = np.searchsorted(np.sort(negatives), grid, side="right") / negatives.size
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _separation_metrics(
    affinities: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Separation / distribution family (threshold-free, non-ranking)."""
    positives = affinities[labels == 1]
    negatives = affinities[labels == 0]

    if positives.size == 0 or negatives.size == 0:
        return {
            "mean_separation": float("nan"),
            "cohens_d": float("nan"),
            "ks_statistic": float("nan"),
        }

    mean_pos = float(np.mean(positives))
    mean_neg = float(np.mean(negatives))
    mean_separation = mean_pos - mean_neg

    # Pooled standard deviation for Cohen's d (standardized effect size).
    n1, n2 = positives.size, negatives.size
    var_pos = float(np.var(positives, ddof=1)) if n1 > 1 else 0.0
    var_neg = float(np.var(negatives, ddof=1)) if n2 > 1 else 0.0
    if (n1 + n2 - 2) > 0:
        pooled_var = ((n1 - 1) * var_pos + (n2 - 1) * var_neg) / (n1 + n2 - 2)
    else:
        pooled_var = 0.0
    pooled_std = float(np.sqrt(pooled_var))

    return {
        "mean_separation": mean_separation,
        "cohens_d": (mean_separation / pooled_std) if pooled_std > 0 else float("nan"),
        "ks_statistic": _ks_statistic(positives, negatives),
    }


def evaluate_groups(
    group_scores: Sequence[GroupObservationScores],
    aggregator: Callable[[np.ndarray], float],
    *,
    aggregator_name: Optional[str] = None,
    min_score_to_consider: float = 0.1,
    top_n: Optional[int] = None,
    decision_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Aggregate each group to one affinity, then score it against the labels.

    Produces a single flat dict containing all three metric families (ranking,
    threshold/classification, and separation/distribution) plus some metadata,
    so you can tune for whichever family matters most to you.

    Args:
        group_scores: Output of :func:`score_groups`.
        aggregator: A summarize metric such as ``skewness`` or ``top_k_mean``.
        aggregator_name: Optional label for the aggregator (defaults to its
            ``__name__``).
        min_score_to_consider: Per-observation threshold applied before
            aggregating (values below it are set to 0).
        top_n: Size of the top-N slice for recall@N / precision@N. Defaults to
            the number of positive groups.
        decision_threshold: Cutoff for the classification family. If ``None``,
            the best-F1 cutoff is found automatically.

    Returns:
        A dict with keys for metadata (``aggregator``, ``min_score_to_consider``,
        ``n_groups``, ``n_positive``) and every metric from the three families.
    """
    labels_list: List[int] = []
    affinities_list: List[float] = []
    for group in group_scores:
        thresholded = _apply_threshold(
            group.observation_scores, min_score_to_consider
        )
        affinity = float(aggregator(thresholded)) if thresholded.size > 0 else 0.0
        labels_list.append(int(group.label))
        affinities_list.append(affinity)

    labels = np.asarray(labels_list, dtype=int)
    affinities = np.asarray(affinities_list, dtype=float)

    row: Dict[str, float] = {
        "aggregator": aggregator_name
        or getattr(aggregator, "__name__", str(aggregator)),
        "min_score_to_consider": float(min_score_to_consider),
        "n_groups": int(labels.size),
        "n_positive": int(np.sum(labels == 1)),
    }
    row.update(_ranking_metrics(affinities, labels, top_n=top_n))
    row.update(_threshold_metrics(affinities, labels, decision_threshold=decision_threshold))
    row.update(_separation_metrics(affinities, labels))
    return row


def compare_aggregators(
    group_scores: Sequence[GroupObservationScores],
    *,
    aggregators: Optional[Mapping[str, Callable[[np.ndarray], float]]] = None,
    min_score_to_consider: float = 0.1,
    top_n: Optional[int] = None,
    decision_threshold: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Evaluate several summarize metrics on the same pre-computed scores.

    This is cheap: it reuses the per-observation scores from
    :func:`score_groups`, so no re-encoding happens.

    Args:
        group_scores: Output of :func:`score_groups`.
        aggregators: Name -> function map. Defaults to
            :data:`DEFAULT_AGGREGATORS` (all six built-ins).
        min_score_to_consider: Per-observation threshold applied before
            aggregating.
        top_n: Size of the top-N slice for recall@N / precision@N.
        decision_threshold: Cutoff for the classification family (``None`` =
            best-F1).

    Returns:
        One metric dict per aggregator (see :func:`evaluate_groups`).
    """
    if aggregators is None:
        aggregators = DEFAULT_AGGREGATORS

    return [
        evaluate_groups(
            group_scores,
            fn,
            aggregator_name=name,
            min_score_to_consider=min_score_to_consider,
            top_n=top_n,
            decision_threshold=decision_threshold,
        )
        for name, fn in aggregators.items()
    ]


def _row_count(embeddings: object) -> Optional[int]:
    """Return the number of rows in an embedding tensor, or None if unavailable.

    Args:
        embeddings: A tensor-like object, or None.

    Returns:
        The row count, or None when the object has no usable shape.
    """
    shape = getattr(embeddings, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[0])


def run_grid_search(
    index: "SentinelLocalIndex",
    groups: Sequence[LabeledGroup],
    *,
    # What to sweep, outermost axis first, mirroring the loop nesting below.
    n_positive_values: Optional[Sequence[int]] = None,
    neg_to_pos_ratios: Optional[Sequence[float]] = None,
    top_k_values: Sequence[int] = (5,),
    min_score_values: Sequence[float] = (0.1,),
    aggregators: Optional[Mapping[str, Callable[[np.ndarray], float]]] = None,
    # How each configuration is judged.
    top_n: Optional[int] = None,
    decision_threshold: Optional[float] = None,
    # Execution.
    index_seed: Optional[int] = None,
    cache_observation_embeddings: bool = True,
    show_progress_bar: bool = False,
) -> List[Dict[str, float]]:
    """Sweep hyperparameters and summarize metrics, returning a flat result table.

    The sweep is organised around what each setting costs. Index size and ratio
    reshape the index (cheap, via
    :meth:`~sentinel.sentinel_local_index.SentinelLocalIndex.subsample`), ``top_k``
    forces a re-scoring (expensive), and thresholds and aggregators are evaluated
    for free on the cached scores::

        for n_positive in n_positive_values:      # cheap: subsample()
          for ratio in neg_to_pos_ratios:         # cheap: subsample()
            for top_k in top_k_values:            # EXPENSIVE: re-scores
              for min_score in min_score_values:  # cheap
                for aggregator in aggregators:    # cheap

    The returned rows are easy to turn into a ``pandas.DataFrame`` for plotting.

    Cost: the sweep runs
    ``len(n_positive_values) x len(neg_to_pos_ratios) x len(top_k_values)`` scoring
    passes, so a 7x7 size/ratio grid with three ``top_k`` values is 147 of them. The
    observations are encoded once up front and reused across all of them, because an
    observation's embedding depends on the encoder rather than on the index being
    varied. Each pass is then a nearest-neighbour search rather than a re-encode.

    Args:
        index: A loaded :class:`~sentinel.sentinel_local_index.SentinelLocalIndex`.
        groups: The labeled groups to evaluate.
        n_positive_values: Index sizes to try, as counts of positive examples.
            ``None`` (the default) means one pass using the index exactly as given.
        neg_to_pos_ratios: Negative-to-positive ratios to try. ``None`` (the
            default) means one pass using the index exactly as given.
        top_k_values: ``top_k`` values to try (each triggers a re-scoring).
        min_score_values: Per-observation thresholds to try (evaluated cheaply).
        aggregators: Name -> function map. Defaults to :data:`DEFAULT_AGGREGATORS`.
        top_n: Size of the top-N slice for recall@N / precision@N.
        decision_threshold: Cutoff for the classification family (``None`` =
            best-F1).
        index_seed: Seed for the subsampling, so each index configuration is
            reproducible across runs.
        cache_observation_embeddings: Whether to encode the observations once and reuse
            them for every pass. Leave it on unless memory is tight: the cache holds one
            embedding per observation, so a very large evaluation set can be sizeable.
        show_progress_bar: Whether to show the encoder progress bar.

    Returns:
        A list of metric dicts, one per ``(n_positive, neg_to_pos_ratio, top_k,
        min_score_to_consider, aggregator)`` combination. Alongside everything
        :func:`evaluate_groups` returns, each row carries ``top_k`` and four
        ``index_``-prefixed columns describing the index it was scored against: the
        requested ``index_n_positive`` and ``index_neg_to_pos_ratio``, and the actual
        ``index_n_positive_actual`` / ``index_n_negative_actual`` counts. The actual
        counts matter because a request is clipped when the index is smaller than
        asked for - without them two rows can look identical while describing
        different indices.

        The prefix is not decoration. ``evaluate_groups`` already returns an
        ``n_positive`` holding the number of positive *groups* in the evaluation set,
        so an unprefixed index size would overwrite it and quietly turn evaluation
        metadata into an index size.
    """
    if aggregators is None:
        aggregators = DEFAULT_AGGREGATORS

    # None means "one pass, use the index exactly as given", which reproduces the
    # behaviour from before these axes existed.
    positive_options: Sequence[Optional[int]] = (
        list(n_positive_values) if n_positive_values is not None else [None]
    )
    ratio_options: Sequence[Optional[float]] = (
        list(neg_to_pos_ratios) if neg_to_pos_ratios is not None else [None]
    )
    total_configurations = len(positive_options) * len(ratio_options)

    # Hoisted out of every loop below: subsampling changes which rows the index holds,
    # never the encoder, so one set of observation embeddings serves the whole sweep.
    cached_embeddings: Optional[Dict[str, np.ndarray]] = None
    if cache_observation_embeddings:
        cached_embeddings = encode_observations(
            index, groups, show_progress_bar=show_progress_bar
        )

    rows: List[Dict[str, float]] = []
    configuration = 0
    for n_positive in positive_options:
        for ratio in ratio_options:
            configuration += 1

            if n_positive is None and ratio is None:
                # Never call subsample() in the default case, so this keeps working
                # with any index-like object and stays byte-identical to before.
                working_index = index
            else:
                working_index = index.subsample(
                    n_positive=n_positive,
                    neg_to_pos_ratio=ratio,
                    seed=index_seed,
                )

            n_positive_actual = _row_count(
                getattr(working_index, "positive_embeddings", None)
            )
            n_negative_actual = _row_count(
                getattr(working_index, "negative_embeddings", None)
            )

            if total_configurations > 1:
                # A long sweep is otherwise indistinguishable from a hang.
                LOG.info(
                    "Grid search index configuration %d/%d: n_positive=%s ratio=%s "
                    "(actual %s positives / %s negatives)",
                    configuration,
                    total_configurations,
                    n_positive,
                    ratio,
                    n_positive_actual,
                    n_negative_actual,
                )

            for top_k in top_k_values:
                scored = score_groups(
                    working_index,
                    groups,
                    top_k=top_k,
                    observation_embeddings=cached_embeddings,
                    show_progress_bar=show_progress_bar,
                )
                for min_score in min_score_values:
                    for name, fn in aggregators.items():
                        row = evaluate_groups(
                            scored,
                            fn,
                            aggregator_name=name,
                            min_score_to_consider=min_score,
                            top_n=top_n,
                            decision_threshold=decision_threshold,
                        )
                        _add_columns(
                            row,
                            **{
                                COLUMN_TOP_K: int(top_k),
                                COLUMN_INDEX_N_POSITIVE: n_positive,
                                COLUMN_INDEX_NEG_TO_POS_RATIO: ratio,
                                COLUMN_INDEX_N_POSITIVE_ACTUAL: n_positive_actual,
                                COLUMN_INDEX_N_NEGATIVE_ACTUAL: n_negative_actual,
                            },
                        )
                        rows.append(row)

    return rows
