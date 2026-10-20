"""
Predictions are thresholded, split into 26-neighbour connected components,
and ranked by the sum of the original voxel probabilities.  The highest
ranked component is evaluated against the nearest allowed ground-truth pocket;
the same ground-truth instance is used for both DCC and DVO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def load_equivalent_pockets(
    path: str | Path,
    expected_keys: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load and validate the repository-bundled equivalent-pocket artifact."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Equivalent-pocket artifact not found: {path}")

    with np.load(path, allow_pickle=True) as loaded:
        required = {"keys", "offsets", "indices32"}
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        artifact = {name: loaded[name] for name in loaded.files}

    keys = artifact["keys"].astype(str)
    offsets = np.asarray(artifact["offsets"], dtype=np.int64)
    if len(offsets) != len(keys) + 1 or offsets[0] != 0:
        raise ValueError(f"Invalid equivalent-pocket offsets in {path}")
    if offsets[-1] != len(artifact["indices32"]):
        raise ValueError(f"Equivalent-pocket offsets/data mismatch in {path}")
    if len(set(keys.tolist())) != len(keys):
        raise ValueError(f"Duplicate sample keys in {path}")

    if expected_keys is not None:
        expected = np.asarray(expected_keys, dtype=str)
        if not np.array_equal(keys, expected):
            raise ValueError(
                "Equivalent-pocket artifact key/order does not match the "
                "PDBbind2016 evaluation samples"
            )
    return artifact


def select_rank1_component(
    probability: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, int, float, int]:
    """Return the top component mask, component count, score, and label ID."""
    
    probability = np.asarray(probability, dtype=np.float32)
    positive = probability >= threshold
    labels, n_components = ndimage.label(positive, structure=CONNECTIVITY_26)
    if not n_components:
        return np.zeros_like(positive), 0, 0.0, 0

    component_ids = np.arange(1, n_components + 1)
    scores = ndimage.sum(
        probability, labels=labels, index=component_ids,
    )
    best_position = int(np.argmax(scores))
    best_id = int(component_ids[best_position])
    return labels == best_id, int(n_components), float(scores[best_position]), best_id


def _extra_target_masks(
    artifact: Mapping[str, np.ndarray],
    sample_index: int,
    shape: tuple[int, int, int],
) -> list[np.ndarray]:
    if shape != (32, 32, 32):
        raise ValueError(
            "The bundled PDBbind2016 equivalent pockets are defined on 32^3 grids"
        )
    start = int(artifact["offsets"][sample_index])
    stop = int(artifact["offsets"][sample_index + 1])
    targets = []
    for index in range(start, stop):
        target = np.zeros(np.prod(shape), dtype=bool)
        target[np.asarray(artifact["indices32"][index], dtype=np.int64)] = True
        target = target.reshape(shape)
        if target.any():
            targets.append(target)
    return targets


def _match_prediction(
    predicted: np.ndarray,
    targets: Sequence[np.ndarray],
    voxel_size: float,
    worst_dcc: float,
) -> tuple[float, float, int]:
    pred_coords = np.argwhere(predicted)
    if not len(pred_coords):
        return worst_dcc, 0.0, 0

    pred_center = pred_coords.mean(axis=0)
    target_coords = [np.argwhere(target) for target in targets]
    distances = np.asarray([
        np.linalg.norm(pred_center - coords.mean(axis=0)) * voxel_size
        if len(coords) else np.inf
        for coords in target_coords
    ])
    if not np.isfinite(distances).any():
        return worst_dcc, 0.0, 0

    matched = int(np.argmin(distances))
    target = np.asarray(targets[matched], dtype=bool)
    intersection = np.logical_and(predicted, target).sum()
    union = np.logical_or(predicted, target).sum()
    dvo = float(intersection / union) if union else 0.0
    return float(distances[matched]), dvo, matched


def evaluate_rank1_pockets(
    probabilities: np.ndarray,
    truths: np.ndarray,
    keys: Sequence[str],
    *,
    equivalent_artifact: Mapping[str, np.ndarray] | None = None,
    threshold: float = 0.5,
    voxel_size: float = 2.0,
) -> dict[str, np.ndarray]:
    """Evaluate one probability grid per sample with the canonical protocol."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    truths = np.asarray(truths) > 0.5
    keys = np.asarray(keys, dtype=str)
    if probabilities.ndim == 5 and probabilities.shape[1] == 1:
        probabilities = probabilities[:, 0]
    if truths.ndim == 5 and truths.shape[1] == 1:
        truths = truths[:, 0]
    if probabilities.shape != truths.shape or probabilities.ndim != 4:
        raise ValueError(
            f"Expected matching [N,D,H,W] arrays, got "
            f"{probabilities.shape} and {truths.shape}"
        )
    if len(keys) != len(probabilities):
        raise ValueError("Sample key count does not match prediction count")

    artifact_index = None
    if equivalent_artifact is not None:
        artifact_keys = equivalent_artifact["keys"].astype(str)
        artifact_index = {key: index for index, key in enumerate(artifact_keys)}
        missing = [key for key in keys if key not in artifact_index]
        if missing:
            raise ValueError(
                f"Equivalent-pocket artifact is missing {len(missing)} samples; "
                f"first missing key: {missing[0]}"
            )

    grid_shape = tuple(int(value) for value in probabilities.shape[1:])
    worst_dcc = (max(grid_shape) - 1) * voxel_size * np.sqrt(3.0)
    dccs, dvos, empties = [], [], []
    n_components, component_ids, component_scores = [], [], []
    matched_indices = []

    for key, probability, original in zip(keys, probabilities, truths):
        predicted, count, score, component_id = select_rank1_component(
            probability, threshold=threshold,
        )
        targets = [original]
        if equivalent_artifact is not None:
            targets.extend(_extra_target_masks(
                equivalent_artifact, artifact_index[key], grid_shape,
            ))
        dcc, dvo, matched = _match_prediction(
            predicted, targets, voxel_size, worst_dcc,
        )
        dccs.append(dcc)
        dvos.append(dvo)
        empties.append(count == 0)
        n_components.append(count)
        component_ids.append(component_id)
        component_scores.append(score)
        matched_indices.append(matched)

    return {
        "keys": keys,
        "DCC": np.asarray(dccs, dtype=np.float64),
        "DVO": np.asarray(dvos, dtype=np.float64),
        "empty_prediction": np.asarray(empties, dtype=bool),
        "n_prediction_components": np.asarray(n_components, dtype=np.int32),
        "rank1_component_id": np.asarray(component_ids, dtype=np.int32),
        "rank1_component_score": np.asarray(component_scores, dtype=np.float64),
        "matched_gt_index": np.asarray(matched_indices, dtype=np.int32),
    }
