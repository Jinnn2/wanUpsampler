import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
import attention_wan as aw  # noqa: E402


def fake_model_module(log):
    def flash_attention(q, k, v, q_lens=None, k_lens=None, window_size=(-1, -1), **kw):
        log.append(("dense", q.shape[1], k.shape[1]))
        return q.clone()
    return types.SimpleNamespace(flash_attention=flash_attention)


def one_video(mod, L, steps=aw.STEPS, S=2048, text=512):
    q = torch.zeros(1, S, 2, 8)
    c = torch.zeros(1, text, 2, 8)
    for _ in range(steps):
        for _branch in range(2):          # cond, uncond
            for _layer in range(L):
                mod.flash_attention(q=q, k=q, v=q, k_lens=torch.tensor([S]), window_size=(-1, -1))
                mod.flash_attention(q, c, c, k_lens=torch.tensor([text]))  # cross-attention stays dense


class PatchTest(unittest.TestCase):
    def test_dense_counts_and_passes_cross_attention(self):
        log = []
        mod = fake_model_module(log)
        patch = aw.SelfAttentionPatch(mod, 3)
        patch.start("DENSE")
        one_video(mod, 3)
        self.assertEqual(patch.finish(), {"self_attention_calls": aw.STEPS * 2 * 3, "sparse_calls": 0, "mean_kept_fraction": None})
        self.assertEqual(sum(1 for e in log if e[2] == 512), aw.STEPS * 2 * 3)

    def test_svg2_dense_warmup_steps_and_layer_order(self):
        log, seen = [], []
        mod = fake_model_module(log)
        patch = aw.SelfAttentionPatch(mod, 3)

        class Fake:
            def __init__(self, layer):
                self.layer = layer

            def __call__(self, q, k, v, *a):
                seen.append(self.layer)
                return q, 0.25
        patch.kern = patch.flashinfer = object()  # skip the real imports
        patch.start("SVG2")
        patch.layers = {i: Fake(i) for i in range(3)}
        one_video(mod, 3)
        stats = patch.finish()
        dense_steps = int(aw.SVG2_OFFICIAL["first_times_fp"] * aw.STEPS)
        self.assertEqual(stats["sparse_calls"], (aw.STEPS - dense_steps) * 2 * 3)
        self.assertEqual(seen[:6], [0, 1, 2, 0, 1, 2])
        self.assertAlmostEqual(stats["mean_kept_fraction"], 0.25)

    def test_wrong_call_count_fails(self):
        mod = fake_model_module([])
        patch = aw.SelfAttentionPatch(mod, 3)
        patch.start("DENSE")
        one_video(mod, 2)
        with self.assertRaises(RuntimeError):
            patch.finish()

    def test_start_resets_state_per_video(self):
        mod = fake_model_module([])
        patch = aw.SelfAttentionPatch(mod, 2)
        patch.start("DENSE")
        one_video(mod, 2)
        patch.finish()
        patch.layers = {0: "stale"}
        patch.start("DENSE")
        self.assertEqual((patch.calls, patch.layers), (0, {}))



class FinalizeTest(unittest.TestCase):
    """finalize writes a root that the scoring pipeline's published_rows() accepts, with FULL50 reused."""

    def test_finalize_reads_back_through_published_rows(self):
        import csv
        import json
        import tempfile
        from types import SimpleNamespace
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from UNIV_adaptor.scripts.data.score_controlled_factor_requirements import published_rows
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prompts = [{"id": f"cf00{i}", "prompt": f"prompt {i}", "origin": "controlled_factor_v1", "family_id": "f0",
                        "split": "train", "motion": "high", "detail": "low"} for i in range(2)]
            cfg = {"prompts": prompts, "seeds": [42], "evaluation": {"quality_dimensions": ["a"], "diagnostic_dimensions": []}}
            ref, out = tmp / "platform", tmp / "attn"
            (ref / "metrics").mkdir(parents=True)
            (out / "records").mkdir(parents=True)
            recs = []
            for p in prompts:
                for arm in ("FULL50", "SVG2"):
                    video = tmp / f"{p['id']}_{arm}.mp4"
                    video.write_bytes(f"{p['id']}{arm}".encode())
                    job = {"id": aw.job_id(p["id"], 42, arm), "group_id": f"{p['id']}_s42", "prompt_id": p["id"],
                           "prompt": p["prompt"], "seed": 42, "arm": arm if arm != "FULL50" else {"id": "FULL50"}}
                    rec = {"job": job, "video_path": str(video), "video_sha256": aw.sha256(video),
                           "runtime": {"pipeline_seconds": 100.0 if arm == "FULL50" else 90.0},
                           "attention": {"mean_kept_fraction": 0.2}}
                    if arm == "FULL50":
                        recs.append(rec)
                    else:
                        (out / "records" / f"{job['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
            (ref / "plan.json").write_text(json.dumps({"config": cfg, "plan_sha256": "p"}), encoding="utf-8")
            (ref / "dataset_manifest.json").write_text(json.dumps({"plan_sha256": "p", "partial_exploratory": False, "records": recs}),
                                                       encoding="utf-8")
            config = tmp / "cfg.json"
            config.write_text(json.dumps(cfg), encoding="utf-8")
            aw.finalize(SimpleNamespace(out=str(out), config=str(config), reference=str(ref), arms=["SVG2"]))
            manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["records"]), 4)
            # the VBench CSV that `score` writes, faked here
            (out / "metrics").mkdir()
            dims = ("subject_consistency", "background_consistency", "motion_smoothness", "aesthetic_quality",
                    "imaging_quality", "dynamic_degree", "overall_consistency")
            with (out / "metrics" / "quality_by_video.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["observation_id", "video_sha256", "pipeline_seconds", "vbench5", *dims])
                w.writeheader()
                for r in manifest["records"]:
                    w.writerow({"observation_id": r["job"]["id"], "video_sha256": r["video_sha256"],
                                "pipeline_seconds": r["runtime"]["pipeline_seconds"], "vbench5": 0.8, **{d: 0.8 for d in dims}})
            rows, _, quality = published_rows(out)
            cases = sorted({r["case_id"] for r in rows})
            self.assertEqual(len(rows), 4)
            self.assertEqual(cases[0] != "FULL50", True)  # FULL50 is renamed to the shared FULL case id
            self.assertIn("SVG2", cases)
            self.assertEqual({r["group_id"] for r in rows}, {"main_p000_s42", "main_p001_s42"})
            # finalize is idempotent and refuses a changed video
            aw.finalize(SimpleNamespace(out=str(out), config=str(config), reference=str(ref), arms=["SVG2"]))
            (tmp / "cf000_SVG2.mp4").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                aw.finalize(SimpleNamespace(out=str(out), config=str(config), reference=str(ref), arms=["SVG2"]))



FAKE_WORKER = r'''
import json, sys, time, pathlib
a = sys.argv[1:]
arg = lambda k: a[a.index(k) + 1]
out, jobs = pathlib.Path(arg("--out")), json.loads(pathlib.Path(arg("--jobs")).read_text())
print(f"[gpu {arg('--gpu')}] START warmup", flush=True)
for j in jobs:
    print(f"[gpu {arg('--gpu')}] START {j['id']}", flush=True)
    marker = out / ("hung_" + j["id"])
    if j["id"].endswith(out.joinpath("HANG").read_text()) and (not marker.exists() or "ALWAYS" in j["id"]):
        marker.write_text("x")
        time.sleep(600)
    (out / "records").mkdir(exist_ok=True)
    (out / "records" / (j["id"] + ".json")).write_text("{}")
'''


class WatchdogTest(unittest.TestCase):
    def _run(self, tmp, hang_suffix, prompts):
        import json
        import subprocess
        from types import SimpleNamespace
        from unittest import mock
        tmp = Path(tmp)
        (tmp / "HANG").write_text(hang_suffix)
        fake = tmp / "fake_worker.py"
        fake.write_text(FAKE_WORKER)
        cfg = tmp / "cfg.json"
        cfg.write_text(json.dumps({"prompts": [{"id": p, "prompt": p} for p in prompts], "seeds": [42]}))
        real_popen = subprocess.Popen

        def popen(cmd, **kw):
            return real_popen([sys.executable, str(fake), *cmd[3:]], **kw)
        ok = mock.Mock(returncode=0)
        args = SimpleNamespace(config=str(cfg), out=str(tmp), arms=["SPARGE50"], gpus="0,1", prompt_ids=None, first=None,
                               seeds=None, python=sys.executable, model_root="m", hang_seconds=2, max_hangs=2,
                               max_restarts=5, poll_seconds=0.5)
        with mock.patch.object(aw.subprocess, "Popen", popen), mock.patch.object(aw.subprocess, "run", return_value=ok):
            with self.assertRaises(SystemExit) as stop:
                aw.run(args)
        return stop.exception.code, json.loads((tmp / "hangs.json").read_text()) if (tmp / "hangs.json").exists() else {}

    def test_hung_job_is_retried_on_the_same_gpu(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            code, hangs = self._run(tmp, "p1_s42_SPARGE50", ["p0", "p1", "p2", "p3"])
            self.assertEqual(code, 0)
            self.assertEqual(hangs, {"pilot_p1_s42_SPARGE50": 1})
            self.assertEqual(len(list(Path(tmp, "records").glob("*.json"))), 4)

    def test_job_that_always_hangs_is_skipped_others_finish(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            code, hangs = self._run(tmp, "ALWAYS_s42_SPARGE50", ["p0", "ALWAYS", "p2", "p3"])
            self.assertEqual(code, 1)
            self.assertEqual(hangs, {"pilot_ALWAYS_s42_SPARGE50": 2})
            self.assertEqual(len(list(Path(tmp, "records").glob("*.json"))), 3)


if __name__ == "__main__":
    unittest.main()
