"""CPU safety/protocol tests; mocked VBench is not an actual GPU score run."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tarfile

from UNIV_adaptor.scripts.data import evaluate_flashvsr_spatial_baseline as e


class SpatialEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.out = self.root / "evaluation"
        self.cfg = e.read(e.CONFIG)
        self.args = SimpleNamespace(source=self.source, out=self.out, config=e.CONFIG,
                                    ratings=self.out / "ratings", ffprobe=None,
                                    vbench_root=self.root / "vbench", vbench_python="python", ngpus=8)
        pairs = [{"id": f"p0{p}_s{s}_S_B025", "prompt_id": f"p0{p}", "seed": s, "arm": "S_B025",
                  "prompt": f"A test scene {p}. </script><script>alert(1)</script>",
                  "frames": 33, "width": 832, "height": 480, "fps": 16, "lr_width": 416, "lr_height": 240}
                 for p in range(4) for s in (42, 3407)]
        self.original = e.sealed({"schema": "flashvsr_asset_diagnostic_plan_v1",
                                 "config": {"prompt_ids": [f"p0{p}" for p in range(4)], "seeds": [42, 3407],
                                            "spatial_arms": ["S_B025"], "frames": 33, "color_fix": True},
                                 "pairs": pairs}, "plan_sha256")
        e.write_new(self.source / "plan.json", self.original)
        for pair in pairs:
            assets = {}
            for track in e.TRACKS + ("HR_DOWN4", "NATIVE_LR"):
                file = self.source / "media" / pair["id"] / (track + ".mp4")
                e.write_text_new(file, f"fixture-video-bytes-{pair['id']}-{track}")
                assets[track] = {"shape": [33, 480, 832, 3], "fps": 16,
                                 "npz": {"sha256": "missing-exported-npz-" + track},
                                 "video": {"path": f"/old/remote/media/{pair['id']}/{track}.mp4",
                                           "bytes": file.stat().st_size, "sha256": e.file_hash(file)}}
            prepared = {"plan_sha256": self.original["plan_sha256"], "pair_id": pair["id"], "assets": assets}
            e.write_new(self.source / "prepared" / (pair["id"] + ".json"), prepared)
            for kind in ("HR_DOWN4", "NATIVE_LR"):
                width, height = (208, 120) if kind == "HR_DOWN4" else (416, 240)
                receipt = {"plan_sha256": self.original["plan_sha256"], "pair_id": pair["id"], "kind": kind,
                           "input_npz_sha256": assets[kind]["npz"]["sha256"], "asset": assets[kind + "_FLASH"],
                           "sr_preprocess_model_postprocess_seconds": 5.,
                           "timing": {"sr_model_seconds": 3., "actual_sparse_attention_calls": 90,
                                      "successful_color_correction_calls": 1,
                                      "padding": e.diagnostic.padding_spec(width, height, 33)}}
                e.write_new(self.source / "results" / f"{pair['id']}_{kind}.json", receipt)
        self.probe_patch = patch.object(e, "probe", return_value={"frames": 33, "width": 832, "height": 480, "fps": 16})
        self.probe_patch.start()
        self.addCleanup(self.probe_patch.stop)

    def frozen(self, with_blind=False):
        e.plan(self.args)
        e.check(self.args)
        if with_blind:
            e.blind(self.args)
        return e.load(self.args)

    def mutate_json(self, path, function):
        body = e.read(path)
        function(body)
        path.write_text(json.dumps(body), encoding="utf-8")

    def ratings(self, trial, preference="A", rater="alice", exposure=False):
        return {"schema": "flashvsr_spatial_blind_ratings_v1", "bundle_id": e.read(self.out / "blind/study.json")["bundle_id"],
                "rater_id": rater, "prior_exposure": exposure,
                "ratings": [{"id": trial["id"], "A": {a: 4 for a in e.AXES}, "B": {a: 2 for a in e.AXES},
                             "A_flags": [], "B_flags": ["blur"], "preference": preference, "notes": "clear artifact"}]}

    def store_scores(self, p):
        dims = p["config"]["quality_dimensions"] + p["config"]["diagnostic_dimensions"]
        data = {"plan_sha256": p["plan_sha256"], "vbench_identity": {"git_commit": self.cfg["expected_vbench_commit"]},
                "scores": {r["clip_id"]: {d: .9 for d in dims} for r in p["clips"]}, "provenance": {}}
        e.write_new(self.out / "scores.json", e.sealed(data, "scores_sha256"))

    def test_archive_paths_no_npz_models_and_no_source_writes(self):
        before = {p.relative_to(self.source).as_posix(): e.file_hash(p) for p in self.source.rglob("*") if p.is_file()}
        p = self.frozen(True)
        self.assertEqual(len(p["clips"]), 40)
        self.assertEqual({r["main_spatial_density"] for r in p["clips"]}, {.25})
        e.export(self.args, True)
        after = {p.relative_to(self.source).as_posix(): e.file_hash(p) for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_staging_absolute_prompt_map_and_original_bytes(self):
        p = self.frozen()
        mapping = e.read(self.out / "prompt_map.json")
        self.assertEqual(len(mapping), 40)
        for row in p["clips"]:
            path = (self.out / "inputs" / (row["clip_id"] + ".mp4")).resolve()
            self.assertEqual(mapping[str(path)], row["prompt"])
            self.assertEqual(e.file_hash(path), row["video"]["sha256"])

    def test_relocated_source_produces_identical_plan_and_bundle(self):
        p = self.frozen(True)
        relocated = copy.copy(self.args)
        relocated.source = self.root / "downloaded_archive"
        relocated.out = self.root / "another_machine_evaluation"
        shutil.copytree(self.source, relocated.source)
        e.plan(relocated)
        e.check(relocated)
        e.blind(relocated)
        self.assertEqual(e.load(relocated)["plan_sha256"], p["plan_sha256"])
        self.assertEqual(e.read(relocated.out / "blind/study.json"), e.read(self.out / "blind/study.json"))

    def test_all_pairs_balanced_and_repeats_reversed(self):
        self.frozen(True)
        private = e.read(self.out / "blind_private.json")
        primary = {r["id"]: r for r in private["trials"] if not r["repeat_of"]}
        self.assertEqual(len(primary), 32)
        for c in self.cfg["comparisons"]:
            selected = [r for r in primary.values() if r["comparison"] == c["id"]]
            self.assertEqual(sum(r["A"] == r["left"] for r in selected), 4)
        repeated = [r for r in private["trials"] if r["repeat_of"]]
        self.assertEqual(len(repeated), 4)
        for row in repeated:
            original = primary[row["repeat_of"]]
            self.assertEqual((row["A"], row["B"]), (original["B"], original["A"]))

    def test_public_bundle_no_methods_scores_or_seed(self):
        self.frozen(True)
        public = e.read(self.out / "blind/study.json")
        self.assertEqual(len(public["trials"]), 36)
        for row in public["trials"]:
            self.assertEqual(set(row), {"id", "prompt", "A", "B"})
            for side in ("A", "B"):
                self.assertRegex(row[side], r"^media/[0-9a-f]{24}\.mp4$")
        html = (self.out / "blind/index.html").read_text(encoding="utf-8")
        self.assertNotIn('</script><script>alert(1)', html)
        self.assertIn(r"\u003c/script>", html)
        self.assertNotIn("fetch(", html)
        self.assertNotIn("cdn", html.lower())

    @unittest.skipUnless(shutil.which("node"), "Node is optional for offline UI smoke test")
    def test_offline_ui_navigation_autosave_and_export(self):
        self.frozen(True)
        script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[1],'utf8');
const study=html.match(/<script id="study" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements={},storage=new Map();let downloads=0;
function element(){return {value:'',checked:false,textContent:'',append(...items){for(const item of items)if(item.id)elements[item.id]=item;},pause(){},play(){return Promise.resolve();},click(){downloads++;}};}
const document={getElementById(id){if(!elements[id])elements[id]=element();return elements[id];},createElement:element};
document.getElementById('study').textContent=study;
const context={document,localStorage:{setItem:(k,v)=>storage.set(k,v),getItem:k=>storage.get(k)},console,
Blob,URL:{createObjectURL:()=> 'blob:offline',revokeObjectURL:()=>{}},setTimeout:()=>0,assert};
vm.runInNewContext(script+`
$('rater').value='test_rater';$('exposure').value='false';
for(const side of ['A','B'])for(const axis of study.axes)$(side+'_'+axis).value=side==='A'?'4':'2';
$('preference').value='A';$('notes').value='A clear, B blurred';$('next').onclick();
assert.equal(index,1);assert.equal(Object.values(records).filter(complete).length,1);
$('prev').onclick();assert.equal($('A_sharpness').value,'4');assert.equal($('notes').value,'A clear, B blurred');
$('download').onclick();assert.ok($('status').textContent.includes('已导出 1 组'));
`,context);
assert.equal(downloads,1);assert.ok(storage.size>0);
console.log('Offline UI navigation, autosave and export OK');
'''
        result = subprocess.run([shutil.which("node"), "-e", script, str(self.out / "blind/index.html")],
                                capture_output=True, text=True, encoding="utf-8", check=True)
        self.assertIn("Offline UI navigation", result.stdout)

    def test_resume_and_exports_are_verified_whitelist(self):
        self.frozen(True)
        e.export(self.args, True)
        e.export(self.args, True)
        exports = list((self.out / "exports").glob("*.tgz"))
        self.assertEqual(len(exports), 1)
        with tarfile.open(exports[0]) as archive:
            names = archive.getnames()
            self.assertEqual(len(names), 43)
            self.assertNotIn("blind_private.json", " ".join(names))
            self.assertTrue(all(n == "manifest.json" or n.startswith("video_review/") for n in names))
        e.plan(self.args)
        e.check(self.args)
        e.blind(self.args)

    def test_changed_source_bytes_rejected(self):
        p = self.frozen()
        path = self.source / p["clips"][0]["relative_video"]
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash/size"):
            e.load(self.args)

    def test_source_plan_bad_seal_rejected(self):
        self.mutate_json(self.source / "plan.json", lambda b: b.update(plan_sha256="bad"))
        with self.assertRaisesRegex(ValueError, "seal"):
            e.plan(self.args)

    def test_overlap_rejected(self):
        for out in (self.source, self.source / "subdirectory", self.source.parent):
            args = copy.copy(self.args)
            args.out = out
            with self.assertRaises(ValueError):
                e.plan(args)

    def test_implementation_change_rejected(self):
        self.frozen()
        with patch.object(e, "implementation", return_value={"changed": "1"}):
            with self.assertRaisesRegex(ValueError, "implementation"):
                e.load(self.args)

    def test_staged_tamper_rejected(self):
        p = self.frozen()
        (self.out / "inputs" / (p["clips"][0]["clip_id"] + ".mp4")).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Staged bytes"):
            e.stage(self.args, p)

    def test_actual_frame_mismatch_rejected(self):
        e.plan(self.args)
        for key, value in (("frames", 81), ("width", 416), ("fps", 24)):
            with patch.object(e, "probe", return_value={"frames": 33, "width": 832, "height": 480, "fps": 16} | {key: value}):
                with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                    e.check(self.args)

    def test_check_required_and_tampered_geometry_rejected(self):
        e.plan(self.args)
        with self.assertRaises(FileNotFoundError):
            e.blind(self.args)
        e.check(self.args)
        def mutate(b):
            next(iter(b["geometry"].values()))["frames"] = 81
        self.mutate_json(self.out / "check.json", mutate)
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            e.blind(self.args)

    def test_prepared_binding_and_sr_backend_rejected(self):
        pair = self.original["pairs"][0]
        path = self.source / "prepared" / (pair["id"] + ".json")
        self.mutate_json(path, lambda b: b.update(plan_sha256="foreign"))
        with self.assertRaisesRegex(ValueError, "Prepared"):
            e.plan(self.args)

    def test_sparse_fallback_rejected(self):
        pair = self.original["pairs"][0]
        path = self.source / "results" / f"{pair['id']}_NATIVE_LR.json"
        self.mutate_json(path, lambda b: b["timing"].update(actual_sparse_attention_calls=0))
        with self.assertRaisesRegex(ValueError, "backend"):
            e.plan(self.args)

    def test_missing_source_video_rejected(self):
        pair = self.original["pairs"][0]
        (self.source / "media" / pair["id"] / "FULL.mp4").unlink()
        with self.assertRaises(FileNotFoundError):
            e.plan(self.args)

    def test_unsafe_identifiers_rejected(self):
        for value in ("../file", "x/y", "", "x\\y", " / "):
            with self.assertRaises(ValueError):
                e.safe_id(value)

    def test_score_coverage_and_invalid_numbers_rejected(self):
        p = self.frozen()
        self.store_scores(p)
        scores = e.load_scores(self.args, p)["scores"]
        for value in (True, float("nan"), -1., 1.2, "0.9"):
            changed = copy.deepcopy(scores)
            next(iter(changed.values()))["imaging_quality"] = value
            with self.assertRaises(ValueError):
                e.validate_scores(changed, p)
        changed = copy.deepcopy(scores)
        changed.pop(next(iter(changed)))
        with self.assertRaisesRegex(ValueError, "coverage"):
            e.validate_scores(changed, p)

    def test_mocked_eight_gpu_scoring_dimensions_and_cache(self):
        p = self.frozen()
        identity = {"git_commit": self.cfg["expected_vbench_commit"]}
        calls = []
        def case(*args):
            calls.append(args)
            dimension = args[5][0]
            return SimpleNamespace(scores={r["clip_id"]: {dimension: .95} for r in p["clips"]}, provenance={"dimension": dimension})
        backend = SimpleNamespace(inspect_vbench_checkout=lambda *args, **kw: identity, score_case_directory=case)
        with patch.object(e, "score_module", return_value=backend):
            e.score(self.args)
            e.score(self.args)
        self.assertEqual(len(calls), 7)
        self.assertTrue(all(args[8] == 8 for args in calls))
        e.report(self.args)
        self.assertIsNone(e.read(self.out / "report.json")["end_to_end_speedup"])

    def test_human_foreign_bundle_duplicate_trials_and_invalid_scores(self):
        self.frozen(True)
        private = e.read(self.out / "blind_private.json")
        valid = self.ratings(private["trials"][0])
        self.assertEqual(e.validate_ratings(valid, private), "alice")
        changed = copy.deepcopy(valid)
        changed["bundle_id"] = "foreign"
        with self.assertRaises(ValueError):
            e.validate_ratings(changed, private)
        for value in (True, 1.5, 0, 6, "4"):
            changed = copy.deepcopy(valid)
            changed["ratings"][0]["A"]["sharpness"] = value
            with self.assertRaises(ValueError):
                e.validate_ratings(changed, private)
        changed = copy.deepcopy(valid)
        changed["ratings"].append(changed["ratings"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            e.validate_ratings(changed, private)

    def test_partial_human_coverage_ab_mapping_and_raw_artifacts(self):
        self.frozen(True)
        private = e.read(self.out / "blind_private.json")
        trial = next(r for r in private["trials"] if not r["repeat_of"] and r["A"] == r["right"])
        data = self.ratings(trial)
        data["ratings"][0]["A"]["detail_correctness"] = None
        e.write_new(self.args.ratings / "alice.json", data)
        e.human_report(self.args)
        report = e.read(next((self.out / "human_reports").glob("*/report.json")))
        row = next(r for r in report["pairs"] if r["id"] == trial["id"])
        self.assertEqual(row["delta_sharpness"], 2)
        self.assertIsNone(row["delta_detail_correctness"])
        self.assertEqual(row["votes_right"], 1)
        self.assertEqual(len(report["pairs"]), 32)
        self.assertFalse(report["coverage_minimum_met"])
        self.assertFalse(report["ready_for_confirmatory_claim"])
        self.assertEqual(report["raw_ratings"][0]["B_flags"], ["blur"])

    def test_repeats_excluded_and_reverse_preferences_aligned(self):
        self.frozen(True)
        private = e.read(self.out / "blind_private.json")
        repeat = next(r for r in private["trials"] if r["repeat_of"])
        primary = next(r for r in private["trials"] if r["id"] == repeat["repeat_of"])
        data = self.ratings(primary, "A")
        data["ratings"] += self.ratings(repeat, "B")["ratings"]
        e.write_new(self.args.ratings / "alice.json", data)
        e.human_report(self.args)
        report = e.read(next((self.out / "human_reports").glob("*/report.json")))
        self.assertTrue(report["reliability"][0]["preference_agrees"])
        self.assertEqual(sum(r["n_raters"] for r in report["pairs"]), 1)

    def test_conflicting_rater_exports_rejected(self):
        self.frozen(True)
        private = e.read(self.out / "blind_private.json")
        trial = private["trials"][0]
        e.write_new(self.args.ratings / "alice.json", self.ratings(trial, "A"))
        e.write_new(self.args.ratings / "alice_conflict.json", self.ratings(trial, "B"))
        with self.assertRaisesRegex(ValueError, "Conflicting exports"):
            e.human_report(self.args)

    def test_blind_private_tamper_rejected(self):
        self.frozen(True)
        def mutate(b):
            b["trials"][0]["A"], b["trials"][0]["B"] = b["trials"][0]["B"], b["trials"][0]["A"]
        self.mutate_json(self.out / "blind_private.json", mutate)
        with self.assertRaisesRegex(ValueError, "Existing output differs"):
            e.export(self.args, True)


if __name__ == "__main__":
    unittest.main()
