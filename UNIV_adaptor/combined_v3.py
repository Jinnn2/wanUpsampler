from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file
from UNIV_adaptor.low_budget_protocol import COMBINED_RECORD_SCHEMA


SCORED_RECORD_SCHEMA = "univ_prompt_budget_scored_trajectory_v3"
SCORE_MANIFEST_SCHEMA = "univ_combined_v3_score_manifest_v1"
SCORED_DATASET_SCHEMA = "univ_combined_v3_scored_dataset_v1"
MERGED_DATASET_SCHEMA = "univ_combined_v3_merged_dataset_v1"
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
)
ALLOWED_SPLITS = ("train", "validation")
EXPECTED_CANDIDATE_COUNT = 9

_ARTIFACT_FIELDS = {
    "video_path",
    "video_sha256",
    "video_bytes",
    "runtime_sidecar_path",
    "runtime_sidecar_sha256",
    "endpoint_state",
    "cost",
    "source_record_path",
    "quality",
}


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_digest(value: Any, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return digest


def validate_artifact(artifact: Any, *, label: str, require_quality: bool) -> None:
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{label} must be an object")
    if not str(artifact.get("video_path", "")).strip():
        raise ValueError(f"{label}.video_path is required")
    validate_digest(artifact.get("video_sha256"), label=f"{label}.video_sha256")
    cost = artifact.get("cost")
    if not isinstance(cost, Mapping):
        raise ValueError(f"{label}.cost must be an object")
    seconds = float(cost.get("pipeline_seconds", 0.0))
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{label}.cost.pipeline_seconds must be positive")
    if require_quality:
        validate_quality(artifact.get("quality"), label=f"{label}.quality")


def validate_quality(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if value.get("profile") != "vbench5_mean_v1":
        raise ValueError(f"{label}.profile must be 'vbench5_mean_v1'")
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(
        QUALITY_DIMENSIONS
    ):
        raise ValueError(f"{label}.dimensions must contain canonical VBench-5")
    numbers = []
    for name in QUALITY_DIMENSIONS:
        number = float(dimensions[name])
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{label}.dimensions.{name} must be in [0, 1]")
        numbers.append(number)
    aggregate = float(value.get("vbench5", float("nan")))
    expected = sum(numbers) / len(numbers)
    if not math.isfinite(aggregate) or not math.isclose(
        aggregate, expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{label}.vbench5 is not the exact five-dimension mean")
    diagnostics = value.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise ValueError(f"{label}.diagnostics must be an object")
    for name, raw in diagnostics.items():
        number = float(raw)
        lower = -1.0 if name == "overall_consistency" else 0.0
        if not math.isfinite(number) or not lower <= number <= 1.0:
            raise ValueError(f"{label}.diagnostics.{name} is out of range")


def candidate_descriptor(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in candidate.items() if key not in _ARTIFACT_FIELDS
    }


def action_catalog(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [candidate_descriptor(item) for item in record["budget_candidates"]]


def validate_generated_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != COMBINED_RECORD_SCHEMA:
        raise ValueError(f"record.schema must be {COMBINED_RECORD_SCHEMA!r}")
    body = {
        key: value
        for key, value in record.items()
        if key not in {"schema", "record_sha256"}
    }
    if canonical_sha256(body) != record.get("record_sha256"):
        raise ValueError("generated combined record hash mismatch")
    if record.get("generation_status") != "generated_unscored":
        raise ValueError("generated combined record must be generated_unscored")
    _validate_record_identity(record)
    validate_artifact(
        record.get("native_teacher"), label="native_teacher", require_quality=False
    )
    candidates = record.get("budget_candidates")
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("combined record requires exactly nine budget candidates")
    if int(record.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("combined record candidate_count must be nine")
    ids = []
    for index, candidate in enumerate(candidates):
        validate_artifact(
            candidate, label=f"budget_candidates[{index}]", require_quality=False
        )
        artifact_id = str(candidate.get("artifact_id", "")).strip()
        if not artifact_id or candidate.get("budget_id") != artifact_id:
            raise ValueError(
                f"budget_candidates[{index}] has invalid artifact identity"
            )
        ids.append(artifact_id)
    if len(ids) != len(set(ids)):
        raise ValueError("combined record candidate artifact ids must be unique")


def validate_scored_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != SCORED_RECORD_SCHEMA:
        raise ValueError(f"record.schema must be {SCORED_RECORD_SCHEMA!r}")
    body = {
        key: value
        for key, value in record.items()
        if key not in {"schema", "record_sha256"}
    }
    if canonical_sha256(body) != record.get("record_sha256"):
        raise ValueError("scored record hash mismatch")
    if record.get("generation_status") != "scored_vbench5":
        raise ValueError("scored record status must be scored_vbench5")
    _validate_record_identity(record)
    source = record.get("source_record")
    if not isinstance(source, Mapping):
        raise ValueError("scored record requires source_record")
    validate_digest(source.get("file_sha256"), label="source_record.file_sha256")
    validate_digest(source.get("record_sha256"), label="source_record.record_sha256")
    validate_artifact(
        record.get("native_teacher"), label="native_teacher", require_quality=True
    )
    candidates = record.get("budget_candidates")
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("scored record requires exactly nine budget candidates")
    if int(record.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("scored record candidate_count must be nine")
    ids = []
    for index, candidate in enumerate(candidates):
        validate_artifact(
            candidate, label=f"budget_candidates[{index}]", require_quality=True
        )
        artifact_id = str(candidate.get("artifact_id", ""))
        if candidate.get("budget_id") != artifact_id:
            raise ValueError(
                f"budget_candidates[{index}] has invalid artifact identity"
            )
        ids.append(artifact_id)
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("scored candidate artifact ids must be non-empty and unique")
    provenance = record.get("scoring_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("scored record requires scoring_provenance")
    validate_digest(
        provenance.get("score_manifest_sha256"),
        label="scoring_provenance.score_manifest_sha256",
    )


def verify_file(path: str | Path, expected_sha256: str, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    if sha256_file(resolved) != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch: {resolved}")
    return resolved


def _validate_record_identity(record: Mapping[str, Any]) -> None:
    for field in ("trajectory_key", "split", "prompt", "prompt_sha256", "seed"):
        if field not in record or record[field] in (None, ""):
            raise ValueError(f"record requires {field}")
    if record["split"] not in ALLOWED_SPLITS:
        raise ValueError(f"selection records must use one of {ALLOWED_SPLITS}")
    int(record["prompt_id"])
    int(record["seed"])
    if "\n" in str(record["prompt"]) or "\r" in str(record["prompt"]):
        raise ValueError("record prompt must occupy exactly one prompt-file line")
    if canonical_sha256(str(record["prompt"])) != record["prompt_sha256"]:
        raise ValueError("record prompt_sha256 mismatch")


def quality_payload(
    scores: Mapping[str, float], diagnostic_dimensions: list[str]
) -> dict[str, Any]:
    dimensions = {name: float(scores[name]) for name in QUALITY_DIMENSIONS}
    return {
        "profile": "vbench5_mean_v1",
        "vbench5": sum(dimensions.values()) / len(dimensions),
        "dimensions": dimensions,
        "diagnostics": {name: float(scores[name]) for name in diagnostic_dimensions},
    }
