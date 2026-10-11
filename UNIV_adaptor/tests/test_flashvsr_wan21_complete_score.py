"""CPU/mock-inference tests of full-clip scoring and exact baseline reuse."""
import copy
import csv
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from UNIV_adaptor.scripts.data import score_flashvsr_wan21_complete as scoring
from UNIV_adaptor.tests import test_flashvsr_wan21_complete as generation_tests

diag = scoring.diag
backend = scoring.helpers.score_module()


class CompleteScoringTests(unittest.TestCase):
    def setUp(self):
        class Fixture(generation_tests.CompletionTests):
            def make_source(inner):
                super().make_source()
                plan = diag.read(inner.source / "plan.json")
                plan["config"]["evaluation"] = {"quality_dimensions": scoring.QUALITY,
                                               "diagnostic_dimensions": scoring.DIAGNOSTIC}
                plan["plan_sha256"] = diag.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
                (inner.source / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
                for path in (inner.source / "records").glob("*.json"):
                    row = diag.read(path)
                    row["plan_sha256"] = plan["plan_sha256"]
                    path.write_text(json.dumps(row), encoding="utf-8")
        self.fixture = Fixture("test_plan_resume")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.gen = self.fixture.make_results()
        scoring.generation.finalize(self.fixture.args)
        self.out = self.fixture.root / "scoring"
        self.args = SimpleNamespace(source=self.fixture.out, out=self.out,
                                    vbench_root=self.fixture.root / "vbench", vbench_python="fake-python",
                                    gpu_ids="4,5,6,7", ffprobe=None)
        self.identity = {"git_commit": scoring.VBENCH_COMMIT, "tracked_dirty": False,
                         "tracked_dirty_paths": [], "evaluate_py_sha256": "fixture"}
        self.make_old_scores()

    def raw_run(self, folder, videos, prompt_map, dimensions, quality, diagnostic, value):
        folder.mkdir(parents=True, exist_ok=True)
        request, _ = backend.build_case_request(video_dir=videos, prompt_map=prompt_map,
                    dimensions=dimensions, quality_dimensions=quality, diagnostic_dimensions=diagnostic,
                    python_bin="fake-python", vbench_identity=self.identity)
        diag.write_new(folder / "score_request.json", request)
        scoring.helpers.copy_verified(prompt_map, folder / "prompt_map.json", diag.bound_file(prompt_map))
        result = {d: [value, [{"video_path": str(path), "video_results": value}
                              for path in sorted(videos.glob("*.mp4"))]] for d in dimensions}
        diag.write_new(folder / "fixture_eval_results.json", result)
        diag.write_new(folder / "fixture_full_info.json", [{"fixture": True}])
        manifest = {"schema": backend.CASE_RESULT_SCHEMA, "request_sha256": request["request_sha256"],
                    "request_file": "score_request.json", "result_file": "fixture_eval_results.json",
                    "result_sha256": diag.file_hash(folder / "fixture_eval_results.json"),
                    "full_info_file": "fixture_full_info.json", "full_info_sha256": diag.file_hash(folder / "fixture_full_info.json"),
                    "dimensions": dimensions, "video_count": 24}
        diag.write_new(folder / "score_run_manifest.json", manifest)
        return manifest | {"run_manifest_path": str(folder / "score_run_manifest.json")}

    def make_old_scores(self):
        study = self.fixture.source
        provenance, csv_rows = {}, []
        for arm in scoring.generation.BASE_ARMS:
            folder = study / "metrics/inputs" / arm
            mapping = {}
            for pair in self.gen["pairs"]:
                binding = pair["comparison"][arm]["video"]
                clip_id = f"pilot_{pair['prompt_id']}_s{pair['seed']}_{arm}"
                target = folder / (clip_id + ".mp4")
                scoring.helpers.copy_verified(Path(binding["path"]), target, binding)
                mapping[str(target)] = pair["prompt"]
                csv_rows.append({"observation_id": clip_id, "action_id": arm, "prompt": pair["prompt"],
                                 "seed": pair["seed"], "prompt_key": pair["prompt_id"],
                                 "video_sha256": binding["sha256"]} | {d: .7 for d in scoring.DIMENSIONS})
            prompt_map = folder / "prompt_map.json"
            diag.write_new(prompt_map, mapping)
            provenance[arm] = self.raw_run(study / "metrics/vbench" / arm / "runs/run_fixture", folder, prompt_map,
                                           scoring.DIMENSIONS, scoring.QUALITY, scoring.DIAGNOSTIC, .7)
        csv_path = study / "metrics/quality_by_video.csv"
        scoring.helpers.write_csv(csv_path, csv_rows)
        diag.write_new(study / "metrics/score_provenance.json",
                       {"plan_sha256": self.gen["source_plan_sha256"], "quality_csv_sha256": diag.file_hash(csv_path), "vbench": provenance})

    def frozen(self):
        scoring.plan(self.args)
        with patch.object(scoring.helpers, "probe", return_value={"width": 832, "height": 480, "frames": 81, "fps": 16}):
            scoring.check(self.args)
        return scoring.load(self.args)

    def mocked_score(self):
        calls = []
        def case(*args):
            calls.append((args, os.environ.get("CUDA_VISIBLE_DEVICES")))
            dimension = args[5][0]
            provenance = self.raw_run(args[4] / "runs/new_fixture", args[2], args[3], args[5], args[6], args[7], .8)
            result = backend.parse_vbench_eval_result(Path(provenance["run_manifest_path"]).parent / provenance["result_file"],
                         dimensions=args[5], expected_stems={x.stem for x in args[2].glob("*.mp4")})
            return SimpleNamespace(scores=result, provenance=provenance)
        fake = SimpleNamespace(inspect_vbench_checkout=lambda *a, **k: self.identity,
                               score_case_directory=case, canonical_json_sha256=backend.canonical_json_sha256,
                               parse_vbench_eval_result=backend.parse_vbench_eval_result)
        with patch.object(scoring.helpers, "score_module", return_value=fake):
            scoring.score(self.args)
            scoring.score(self.args)
        return calls

    def test_full_inventory_only24_staged_and_no_source_writes(self):
        original = {str(p): diag.file_hash(p) for p in self.fixture.source.rglob("*") if p.is_file()}
        p = self.frozen()
        self.assertEqual(len(p["clips"]), 120)
        self.assertEqual(len(list((self.out / "inputs").glob("*.mp4"))), 24)
        self.assertTrue(all(scoring.generation.METHOD in path.name for path in (self.out / "inputs").glob("*.mp4")))
        self.assertEqual(original, {str(p): diag.file_hash(p) for p in self.fixture.source.rglob("*") if p.is_file()})

    def test_exact_baseline_raw_reuse(self):
        p = self.frozen()
        scores, provenance, files = scoring.baseline_scores(self.args, p, backend, self.identity)
        self.assertEqual(len(scores), 96)
        self.assertEqual(set(provenance), set(scoring.generation.BASE_ARMS))
        self.assertEqual(len(files), 20)

    def test_four_gpu_new_clips_only_and_cached_complete_scores(self):
        p = self.frozen()
        original_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        calls = self.mocked_score()
        self.assertEqual(len(calls), 7)
        self.assertEqual([args[5][0] for args, _ in calls], scoring.DIMENSIONS)
        self.assertTrue(all(args[8] == 4 and mask == "4,5,6,7" for args, mask in calls))
        self.assertTrue(all(len(list(args[2].glob("*.mp4"))) == 24 for args, _ in calls))
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), original_env)
        scoring.report(self.args)
        self.assertEqual(diag.read(self.out / "report.json")["n_videos"], 120)
        self.assertIsNone(diag.read(self.out / "report.json")["end_to_end_speedup"])
        with (self.out / "flash_vs_methods.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 96)
        self.assertAlmostEqual(float(rows[0]["delta_imaging_quality"]), .1)
        self.assertEqual(len(scoring.load_scores(self.args, p)["scores"]), 120)

    def test_export_small_no_videos_and_verified_resume(self):
        self.frozen()
        self.mocked_score()
        scoring.export(self.args)
        archive = self.out.parent / (self.out.name + "_analysis.tgz")
        with tarfile.open(archive) as bundle:
            names = bundle.getnames()
        self.assertFalse(any(x.endswith((".mp4", ".npz", ".pt")) for x in names))
        self.assertEqual(len([x for x in names if "reused_vbench" in x and x.endswith("eval_results.json")]), 4)
        original = diag.file_hash(archive)
        scoring.export(self.args)
        self.assertEqual(original, diag.file_hash(archive))

    def test_33_frame_check_rejected_before_inference(self):
        scoring.plan(self.args)
        with patch.object(scoring.helpers, "probe", return_value={"width": 832, "height": 480, "frames": 33, "fps": 16}):
            with self.assertRaisesRegex(ValueError, "81-frame"):
                scoring.check(self.args)

    def test_missing_old_csv_fails_plan(self):
        (self.fixture.source / "metrics/quality_by_video.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            scoring.plan(self.args)

    def test_different_vbench_identity_not_mixed(self):
        p = self.frozen()
        with self.assertRaisesRegex(ValueError, "settings/version differ"):
            scoring.baseline_scores(self.args, p, backend, {**self.identity, "evaluate_py_sha256": "changed"})

    def test_old_raw_result_tamper_rejected(self):
        p = self.frozen()
        row = p["baseline_score_provenance"]["vbench"]["FULL50"]
        path = Path(row["run_manifest_path"]).parent / row["result_file"]
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            scoring.baseline_scores(self.args, p, backend, self.identity)

    def test_old_request_video_inventory_tamper_rejected(self):
        p = self.frozen()
        old = p["baseline_score_provenance"]["vbench"]["FULL50"]
        root = Path(old["run_manifest_path"]).parent
        req = diag.read(root / "score_request.json")
        req["videos"][0]["sha256"] = "changed"
        req["request_sha256"] = backend.canonical_json_sha256({k: v for k, v in req.items() if k != "request_sha256"})
        (root / "score_request.json").write_text(json.dumps(req), encoding="utf-8")
        manifest = diag.read(root / "score_run_manifest.json")
        manifest["request_sha256"] = req["request_sha256"]
        (root / "score_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        p["baseline_score_provenance"]["vbench"]["FULL50"]["request_sha256"] = req["request_sha256"]
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            scoring.baseline_scores(self.args, p, backend, self.identity)

    def test_invalid_scores_and_valid_negative_clip_similarity(self):
        p = self.frozen()
        values = {r["clip_id"]: {d: .7 for d in scoring.DIMENSIONS} for r in p["clips"]}
        first = next(iter(values))
        values[first]["overall_consistency"] = -.1
        scoring.validate_scores(values, p)
        for invalid in (float("nan"), True, -1, 2, "0.9"):
            bad = copy.deepcopy(values)
            bad[first]["imaging_quality"] = invalid
            with self.assertRaises(ValueError):
                scoring.validate_scores(bad, p)


if __name__ == "__main__":
    unittest.main()
