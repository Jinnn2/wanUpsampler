import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from UNIV_adaptor.data_protocol import (
    RECORD_SCHEMA, build_collection_plan, canonical_sha256, sha256_file, write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import MANIFEST_SCHEMA
from UNIV_adaptor.scripts.data.score_phase2_dataset import collect, locate_roots, output_lock, stage_inputs, score, main, SCORE_SCHEMA
from UNIV_adaptor.scripts.data.phase2_analysis import (
    DIMENSIONS, NATIVE, aggregate_prompts, budget_analysis, enrich, report, runtime_report,
)

ROOT = Path(__file__).resolve().parents[2]


def fixture(root):
    protocol = json.loads((ROOT / "UNIV_adaptor/configs/univ_prompt_budget_phase2.json").read_text())
    protocol["splits"][0]["prompt_count"] = 2
    protocol["splits"][1]["prompt_count"] = 2
    plan = build_collection_plan(protocol, [f"unique prompt {i}" for i in range(5)])
    write_json_atomic(root / "collection_plan.json", plan)
    body = {"plan_sha256": plan["plan_sha256"], "protocol_sha256": plan["protocol_sha256"],
            "plan_path": str(root / "collection_plan.json")}
    manifest = {**body, "schema": MANIFEST_SCHEMA, "manifest_sha256": canonical_sha256(body)}
    write_json_atomic(root / "generation_manifest.json", manifest)
    for assignment in plan["assignments"]:
        if assignment["split"] == "test":
            continue
        artifacts = []
        for index, action in enumerate([NATIVE] + [c["budget_id"] for c in assignment["budget_candidates"]]):
            path = root / "videos" / (assignment["trajectory_key"] + "__" + action + ".mp4")
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(path.name.encode())
            artifact = dict(video_path=str(path), video_sha256=sha256_file(path), video_bytes=path.stat().st_size,
                            cost={"pipeline_seconds": 100.0 if index == 0 else 10.0 * index})
            if index:
                sidecar = path.with_suffix(".mp4.univ.json")
                sidecar.write_text('{"execution": "fixture"}')
                artifact.update(runtime_sidecar_path=str(sidecar), runtime_sidecar_sha256=sha256_file(sidecar))
            artifacts.append(artifact)
        record = {k: assignment[k] for k in ("trajectory_key", "split", "prompt_id", "prompt", "prompt_sha256", "seed", "base_seed")}
        record.update(schema=RECORD_SCHEMA, generation_status="generated_unscored", plan_sha256=plan["plan_sha256"],
                      native_teacher=artifacts[0], budget_candidates=[{**c, **a} for c, a in zip(assignment["budget_candidates"], artifacts[1:])],
                      provenance={"generation_manifest_sha256": manifest["manifest_sha256"], "protocol_sha256": plan["protocol_sha256"]})
        write_json_atomic(root / "records" / assignment["split"] / (assignment["trajectory_key"] + ".json"), record)
    return plan


def synthetic_scores(rows):
    actions = sorted({r["action_id"] for r in rows if r["action_id"] != NATIVE})
    return {r["stem"]: {d: .9 if r["action_id"] == NATIVE else .5 + .02 * actions.index(r["action_id"]) for d in DIMENSIONS}
            for r in rows}


class Phase2QualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture(self.root)

    def read_record(self):
        path = next((self.root / "records/train").glob("*.json"))
        return path, json.loads(path.read_text())

    def test_collect_real_v2_schema_and_native_optional_sidecar(self):
        rows, identity = collect(self.root)
        self.assertEqual(len(rows), 48)
        self.assertEqual(sum(r["action_id"] == NATIVE for r in rows), 8)
        self.assertIn("plan_sha256", identity)
        # A changed current config/revision is deliberately not consulted.
        self.assertNotIn("record_sha256", self.read_record()[1])

    def test_missing_native_rejected(self):
        path, record = self.read_record()
        record["native_teacher"] = None
        write_json_atomic(path, record)
        with self.assertRaisesRegex(ValueError, "native_teacher"):
            collect(self.root)

    def test_missing_and_extra_records_rejected(self):
        path, record = self.read_record()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "coverage"):
            collect(self.root)
        write_json_atomic(path, record)
        write_json_atomic(path.with_name("extra.json"), record)
        with self.assertRaisesRegex(ValueError, "coverage"):
            collect(self.root)

    def test_locate_distinguishes_partial_and_complete_roots_without_writes(self):
        parent = self.root / "outputs"
        partial = parent / "phase2_old"
        complete = parent / "phase2_chunk25"
        fixture(partial)
        fixture(complete)
        next((partial / "records/train").glob("*.json")).unlink()
        before = {str(p): p.stat().st_mtime_ns for p in parent.rglob("*")}
        with redirect_stdout(io.StringIO()) as output:
            entries = locate_roots(parent, partial)
        by_root = {r["root"]: r for r in entries}
        self.assertFalse(by_root[str(partial)]["coverage_complete"])
        self.assertEqual(by_root[str(partial)]["splits"]["train"]["missing"], 1)
        self.assertTrue(by_root[str(complete)]["coverage_complete"])
        self.assertIn("[selected]", output.getvalue())
        self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in parent.rglob("*")})

    def test_locate_multiple_complete_roots_does_not_pick_one(self):
        parent = self.root / "outputs"
        for name in ("phase2_a", "phase2_b"):
            fixture(parent/name)
        with patch("sys.argv", ["score_phase2_dataset.py", "locate", "--search-root", str(parent)]), \
             redirect_stdout(io.StringIO()) as output:
            main()
        self.assertEqual(output.getvalue().count("record coverage: COMPLETE"), 2)
        self.assertIn("No directories were modified or automatically selected", output.getvalue())

    def test_wrong_schedule_rejected(self):
        path, record = self.read_record()
        record["budget_candidates"][0]["resolved_schedule"] = {}
        write_json_atomic(path, record)
        with self.assertRaisesRegex(ValueError, "differs from plan"):
            collect(self.root)

    def test_hash_changes_rejected(self):
        _, record = self.read_record()
        video = Path(record["native_teacher"]["video_path"])
        original = video.read_bytes()
        video.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "video identity"):
            collect(self.root)
        video.write_bytes(original)
        Path(record["budget_candidates"][0]["runtime_sidecar_path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "sidecar identity"):
            collect(self.root)

    def test_record_provenance_rejected(self):
        path, record = self.read_record()
        record["provenance"]["generation_manifest_sha256"] = "wrong"
        write_json_atomic(path, record)
        with self.assertRaisesRegex(ValueError, "provenance"):
            collect(self.root)

    def test_nonfinite_cost_rejected(self):
        path, record = self.read_record()
        record["native_teacher"]["cost"]["pipeline_seconds"] = float("inf")
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "finite"):
            collect(self.root)

    def test_stage_copy_fallback_and_extra_file(self):
        rows, _ = collect(self.root)
        with patch("os.link", side_effect=OSError("no hardlink")):
            inputs = stage_inputs(rows, self.root)
        self.assertEqual(len(list(inputs.glob("*.mp4"))), 48)
        mapping = json.loads((self.root/"prompt_map.json").read_text())
        self.assertTrue(all(Path(p).is_absolute() for p in mapping))
        (inputs / "extra.mp4").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "unexpected staged"):
            stage_inputs(rows, self.root)

    def test_complete_report_and_seed_aggregation(self):
        rows, _ = collect(self.root)
        scores = synthetic_scores(rows)
        report(rows, scores, self.root)
        runtime_report(rows, self.root)
        with (self.root / "quality_by_prompt.csv").open() as handle:
            prompts = list(csv.DictReader(handle))
        self.assertEqual(len(prompts), 24)
        self.assertEqual({r["seeds"] for r in prompts if r["split"]=="validation"}, {"3"})
        native = [r for r in prompts if r["action_id"] == NATIVE]
        self.assertTrue(all(float(r["delta_native_vbench5"]) == 0 for r in native))
        with (self.root / "seed_stability.csv").open() as handle:
            stability = list(csv.DictReader(handle))
        self.assertEqual(len(stability), 2)
        self.assertTrue(all(r["unanimous_winner"] == "True" for r in stability))
        self.assertIn("not independent timestep skipping", (self.root / "report.md").read_text())

    def test_incomplete_pair_and_scores_rejected(self):
        rows, _ = collect(self.root)
        scores = synthetic_scores(rows)
        with self.assertRaisesRegex(ValueError, "pairing"):
            enrich(rows[1:], scores)
        scores.pop(rows[0]["stem"])
        with self.assertRaisesRegex(ValueError, "coverage"):
            enrich(rows, scores)
        scores = synthetic_scores(rows)
        scores[rows[0]["stem"]][DIMENSIONS[0]] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            enrich(rows, scores)

    def test_negative_semantic_cosine_is_valid_diagnostic(self):
        rows, _ = collect(self.root)
        scores = synthetic_scores(rows)
        scores[rows[0]["stem"]]["overall_consistency"] = -.1
        result = enrich(rows, scores)
        self.assertEqual(result[0]["overall_consistency"], -.1)
        self.assertAlmostEqual(result[0]["vbench5"], .9)

    def test_cost_feasibility_and_no_validation_baseline_tuning(self):
        rows, _ = collect(self.root)
        prompts = aggregate_prompts(enrich(rows, synthetic_scores(rows)))
        # Train favors the cheapest action; validation favors the second cheapest.
        for r in prompts:
            if r["action_id"] != NATIVE:
                r["vbench5"] = .9 if r["pipeline_seconds"] == 10 else .5
                if r["split"] == "validation":
                    r["vbench5"] = .95 if r["pipeline_seconds"] == 20 else .5
        _, summary, details = budget_analysis(prompts, [5, 20])
        no_budget = [r for r in summary if r["budget_seconds"] == 5]
        self.assertTrue(all(r["paired_gain"] is None and r["paired_coverage"]==0 for r in no_budget))
        valid = next(r for r in summary if r["split"]=="validation" and r["budget_seconds"]==20)
        self.assertAlmostEqual(valid["paired_gain"], .45)
        baseline = valid["fixed_action"]
        # A slow fixed action on one validation prompt cannot appear in paired gain.
        for r in prompts:
            if r["split"]=="validation" and r["prompt_id"]==2 and r["action_id"]==baseline:
                r["pipeline_seconds"] = 21
        _, summary, details = budget_analysis(prompts, [20])
        valid = next(r for r in summary if r["split"]=="validation")
        self.assertEqual(valid["fixed_action"], baseline)
        self.assertEqual(valid["paired_coverage"], .5)
        self.assertEqual(valid["fixed_violation_fraction"], .5)
        self.assertTrue(all(r["oracle_seconds"] <= 20 for r in details if r["oracle_feasible"]))

    def test_loso_exposes_unstable_seed_winners(self):
        rows, _ = collect(self.root)
        scores = synthetic_scores(rows)
        actions = sorted({r["action_id"] for r in rows if r["action_id"]!=NATIVE})
        for r in rows:
            if r["split"]=="validation" and r["action_id"]!=NATIVE:
                best = actions[[42,100,2024].index(r["base_seed"])]
                scores[r["stem"]] = {d:.99 if r["action_id"]==best else .1 for d in DIMENSIONS}
        report(rows, scores, self.root)
        with (self.root/"seed_stability.csv").open() as handle:
            stability = list(csv.DictReader(handle))
        self.assertTrue(all(float(r["pairwise_winner_agreement"])==0 for r in stability))
        self.assertTrue(all(float(r["loso_regret_vs_seed_oracle"])>.8 for r in stability))

    def test_lock_is_exclusive_and_cleaned(self):
        with output_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "stale lock"):
                with output_lock(self.root):
                    pass
        self.assertFalse((self.root/".evaluation.lock").exists())

    def test_distributed_scoring_resumes_completed_dimensions(self):
        # Exercise the real strict adapter/cache/parser; only replace GPU execution.
        from changing_resolution_uni.scripts.data import batch_vbench_score_dataset as backend
        rows, _ = collect(self.root)
        out = self.root / "evaluation"
        out.mkdir()
        calls = []
        fail_once = [True]

        def fake_gpu(command, **kwargs):
            self.assertIn("--nproc_per_node=8", command)
            dimension = command[command.index("--dimension")+1]
            calls.append(dimension)
            if dimension == DIMENSIONS[2] and fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("simulated GPU interruption")
            run_dir = Path(command[command.index("--output_path")+1])
            mapping = json.loads(Path(command[command.index("--prompt_file")+1]).read_text())
            write_json_atomic(run_dir/"fixture_eval_results.json", {
                dimension: [.5, [{"video_path": p, "video_results": .5} for p in mapping]]})
            write_json_atomic(run_dir/"fixture_full_info.json", [])

        args = SimpleNamespace(vbench_root=str(self.root), vbench_python="fixture-python", ngpus=8, force_rescore=False)
        with patch.object(backend, "inspect_vbench_checkout", return_value={"fixture": True}), \
             patch.object(backend.subprocess, "run", side_effect=fake_gpu):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                score(rows, out, args, "fixture-digest")
            self.assertFalse((out/"scores.json").exists())
            score(rows, out, args, "fixture-digest")
        self.assertEqual(calls.count(DIMENSIONS[0]), 1)
        self.assertEqual(calls.count(DIMENSIONS[1]), 1)
        self.assertEqual(calls.count(DIMENSIONS[2]), 2)
        self.assertEqual(len(calls), 8)
        payload = json.loads((out/"scores.json").read_text())
        self.assertEqual(len(payload["scores"]), 48)
        self.assertEqual(set(payload["provenance"]), set(DIMENSIONS))
        self.assertTrue(all(set(s)==set(DIMENSIONS) for s in payload["scores"].values()))

    def test_cli_check_report_and_stale_score_guard(self):
        out = self.root/"metrics/phase2_quality"
        argv = ["score_phase2_dataset.py", "check", "--dataset-root", str(self.root)]
        with patch("sys.argv", argv):
            main()
        inputs = json.loads((out/"evaluation_inputs.json").read_text())
        body = dict(schema=SCORE_SCHEMA, input_sha256=inputs["input_sha256"], dimensions=list(DIMENSIONS),
                    scores=synthetic_scores(inputs["rows"]), provenance={"fixture": True})
        write_json_atomic(out/"scores.json", {**body, "payload_sha256":canonical_sha256(body)})
        argv[1] = "report"
        with patch("sys.argv", argv):
            main()
        self.assertTrue((out/"report.md").is_file())
        body["input_sha256"] = "stale"
        write_json_atomic(out/"scores.json", {**body, "payload_sha256":canonical_sha256(body)})
        with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "different inputs"):
            main()


if __name__ == "__main__":
    unittest.main()
