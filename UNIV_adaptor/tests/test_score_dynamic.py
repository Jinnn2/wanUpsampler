import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from UNIV_adaptor.scripts.data import requirements_v2 as rv2  # noqa: E402
from UNIV_adaptor.scripts.data import score_dynamic as sd  # noqa: E402

PROMPTS = [REPO / "prompts/univ_controlled_factor_v1.jsonl", REPO / "prompts/univ_controlled_factor_dev2.jsonl"]
FROZEN = REPO / "prompts/univ_requirements_v2_frozen.jsonl"


class FakeJudge:
    """Answers from the prompt text and the identity of the images, recording every call."""

    def __init__(self):
        self.calls = []

    def p_yes(self, prompts, images=None, videos=None, metadata=None):
        self.calls.append((prompts, images, videos))
        return [0.9 if "B satisfy" in p[-1] else 0.2 for p in prompts]


class OutOfMemoryError(RuntimeError):
    pass


class TightJudge(FakeJudge):
    """Runs out of memory on any batch larger than `limit`."""

    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def p_yes(self, prompts, images=None, videos=None, metadata=None):
        if len(prompts) > self.limit:
            raise OutOfMemoryError("CUDA out of memory")
        self.calls.append((prompts, images, videos))
        return [0.1 * len(p[-1]) % 1 for p in prompts]


class SplitTest(unittest.TestCase):
    def test_oom_halves_the_batch_and_keeps_order(self):
        prompts = [["video", f"q{'x' * i}"] for i in range(7)]
        extra = {"videos": [f"v{i}" for i in range(7)], "metadata": [{"i": i} for i in range(7)]}
        want = TightJudge(99).p_yes(prompts)
        judge = TightJudge(2)
        self.assertEqual(sd.p_yes_split(judge, prompts, extra), want)
        self.assertEqual(sum(len(c[0]) for c in judge.calls), 7)

    def test_other_errors_and_single_prompts_raise(self):
        with self.assertRaises(OutOfMemoryError):
            sd.p_yes_split(TightJudge(0), [["video", "q"]], {"videos": ["v"]})


class DecisionTest(unittest.TestCase):
    def test_gate_propagates_failures_down_dependencies(self):
        qs = [{"id": "q0", "depends_on": []}, {"id": "q1", "depends_on": ["q0"]},
              {"id": "q2", "depends_on": ["q1"]}, {"id": "q3", "depends_on": []}]
        self.assertEqual(sd.gate(qs, {"q0": False, "q1": True, "q2": True, "q3": True}),
                         {"q0": False, "q1": False, "q2": False, "q3": True})

    def test_type_balanced_weights_types_equally(self):
        qs = [{"id": f"q{i}", "type": t} for i, t in enumerate(["attribute"] * 3 + ["static"])]
        verdict = {"q0": True, "q1": True, "q2": True, "q3": False}
        self.assertAlmostEqual(sd.type_balanced(qs, verdict), 0.5)
        self.assertAlmostEqual(sd.type_balanced(qs, verdict, among={"q0", "q3"}), 0.5)

    def test_measured_rules(self):
        still = {"type": "static", "region": "subject", "expect": "still"}
        fast = {"type": "motion", "region": "subject", "expect": "fast"}
        orbit = {"type": "camera", "region": "background", "expect": "moving"}
        self.assertTrue(sd.measured_pass(still, 0.2))
        self.assertFalse(sd.measured_pass(still, 0.8))
        self.assertTrue(sd.measured_pass(fast, 3.0))
        self.assertFalse(sd.measured_pass(fast, 1.0))
        self.assertTrue(sd.measured_kept(still, 0.15, 0.1))     # (0.2 / 0.15) < 2
        self.assertFalse(sd.measured_kept(still, 0.5, 0.1))     # rCM adding motion to a still scene
        self.assertFalse(sd.measured_kept(fast, 1.0, 3.0))      # slow motion keeps a third of FULL's motion
        self.assertTrue(sd.measured_kept(orbit, 2.0, 3.0))
        steady = sd.lighting_stats([100.0] * 81)
        drift = sd.lighting_stats(list(np.linspace(100, 130, 81)))
        self.assertTrue(sd.measured_pass({"type": "lighting"}, steady))
        self.assertFalse(sd.measured_pass({"type": "lighting"}, drift))
        self.assertFalse(sd.measured_kept({"type": "lighting"}, drift, steady))
        self.assertEqual(sd.measure(still, {"flow_local": 0.3, "flow_mag": 1.0, "flow_camera": 0.1}), 0.3)


class PairwiseTest(unittest.TestCase):
    def test_both_orders_ask_about_the_accelerated_clip(self):
        q = {"question": "Is the crane white?"}
        full_first = sd.paired_prompt("image", q, accel_first=False)
        accel_first = sd.paired_prompt("image", q, accel_first=True)
        self.assertIn("Does image B satisfy", full_first[-1])
        self.assertIn("at least as well as image A", full_first[-1])
        self.assertIn("Does image A satisfy", accel_first[-1])
        self.assertEqual([p for p in full_first if p in ("image", "video")], ["image", "image"])

    def test_score_pair_passes_frames_in_the_stated_order(self):
        full = (np.zeros((81, 8, 8, 3), np.uint8), 16.0)
        accel = (np.full((81, 8, 8, 3), 255, np.uint8), 16.0)
        judge = FakeJudge()
        qs = [{"id": "q0", "type": "attribute", "question": "Is it white?"}]
        out = sd.score_pair(judge, accel, full, qs)
        self.assertAlmostEqual(out["q0"]["p_full_first"], 0.9)
        self.assertAlmostEqual(out["q0"]["p_accel_first"], 0.2)
        prompts, images, _ = judge.calls[0]  # FULL first: FULL frame, then the accelerated frame
        self.assertEqual((images[0].max(), images[1].max()), (0, 255))
        prompts, images, _ = judge.calls[1]
        self.assertEqual((images[0].max(), images[1].max()), (255, 0))
        self.assertEqual(len(judge.calls), 2 * sd.FRAME_COUNT)


class ReportTest(unittest.TestCase):
    """plan -> hand-written vlm/pair caches -> report, on one real frozen prompt and two clips."""

    def test_report_end_to_end(self):
        record = next(r for r in rv2.load_jsonl(FROZEN) if r["prompt_id"] == 100)
        prompt = rv2.load_prompts(PROMPTS)[100]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fixed = tmp / "fixed"
            (fixed / "cache").mkdir(parents=True)
            base = {"source": "src", "group_id": "main_p100_s42", "prompt_id": 100, "family_id": prompt["family_id"],
                    "split": prompt["split"], "motion_level": prompt["motion_level"],
                    "detail_level": prompt["detail_level"], "base_seed": 42, "video_path": "unused"}
            videos = [{**base, "stem": "full", "case_id": sd.FULL, "video_sha256": "f" * 64},
                      {**base, "stem": "fast", "case_id": "RCM4", "video_sha256": "a" * 64}]
            (fixed / "plan.json").write_text(json.dumps({"videos": videos}), encoding="utf-8")
            flows = {"f" * 64: 0.1, "a" * 64: 1.2}  # the accelerated clip moves a still crane
            for sha, flow in flows.items():
                (fixed / "cache" / f"{sha}.json").write_text(json.dumps(
                    {"flow_mag": flow, "flow_local": flow, "flow_camera": 0.05, "sharpness_temporal_std": 0.05}))
            out = tmp / "dyn"
            with contextlib.redirect_stdout(io.StringIO()):
                sd.plan(Namespace(fixed_root=str(fixed), checklist=str(FROZEN), prompts=[str(p) for p in PROMPTS],
                                  prompt_ids=[100], out=str(out)))
            value = sd.load_plan(out)
            vq = sd.vlm_questions(record)
            for sha in flows:
                sd.write_json_atomic(out / "vlm" / f"{sha}.json", {
                    "video_sha256": sha, "protocol": sd.PROTOCOL, "checklist_sha256": value["checklist_sha256"],
                    "luma_series": [120.0] * 81, "answers": {q["id"]: {"p_yes": 0.9} for q in vq}})
            value["out"] = str(out)
            (key, (_, _, qs)), = sd.pair_jobs(value)
            self.assertEqual(len(qs), len(vq))  # FULL satisfies every VLM question, so all are asked pairwise
            sd.write_json_atomic(out / "pair" / f"{key}.json", {
                "answers": {q["id"]: {"p_full_first": 0.8, "p_accel_first": 0.7} for q in qs}})
            with contextlib.redirect_stdout(io.StringIO()):
                sd.report(Namespace(out=str(out)))
            rows = list(csv.DictReader((out / "dynamic_by_video.csv").read_text(encoding="utf-8").splitlines()))
            accel = next(r for r in rows if r["case_id"] == "RCM4")
            self.assertAlmostEqual(float(accel["retention_vlm_frames"]), 1.0)
            self.assertLess(float(accel["retention_flow"]), 1.0)  # the static questions were lost
            self.assertLess(float(accel["retention"]), 1.0)
            static = [r for r in csv.DictReader((out / "dynamic_by_question.csv").read_text(encoding="utf-8").splitlines())
                      if r["case_id"] == "RCM4" and r["type"] == "static"]
            self.assertTrue(static and all(r["kept"] == "0" for r in static))


if __name__ == "__main__":
    unittest.main()
