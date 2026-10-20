import argparse
import copy
import json
import unittest
from pathlib import Path

import importlib.util

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("requirements_v2", REPO / "UNIV_adaptor/scripts/data/requirements_v2.py")
rv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv2)

PROMPTS = [REPO / "prompts/univ_controlled_factor_v1.jsonl", REPO / "prompts/univ_controlled_factor_dev2.jsonl"]
PILOT = REPO / "prompts/univ_requirements_v2_pilot.jsonl"


def prompt_index():
    return {p["prompt_id"]: p for path in PROMPTS for p in rv2.load_jsonl(path)}


class RequirementsV2Test(unittest.TestCase):
    def test_pilot_is_valid(self):
        prompts = prompt_index()
        for rec in rv2.load_jsonl(PILOT):
            self.assertEqual(rv2.validate_record(rec, prompts[rec["prompt_id"]]), [], rec["prompt_id"])

    def test_every_type_has_a_judge(self):
        types = {q["type"] for rec in rv2.load_jsonl(PILOT) for q in rec["questions"]}
        self.assertLessEqual(types, set(rv2.JUDGE))

    def test_rejects_bad_records(self):
        prompts = prompt_index()
        rec = rv2.load_jsonl(PILOT)[0]
        p = prompts[rec["prompt_id"]]
        bad = copy.deepcopy(rec)
        bad["questions"][0]["evidence"] = "words that are not in the prompt"
        self.assertTrue(any("evidence" in e for e in rv2.validate_record(bad, p)))
        bad = copy.deepcopy(rec)
        bad["questions"][0]["depends_on"] = ["q5"]
        self.assertTrue(any("depends_on" in e for e in rv2.validate_record(bad, p)))
        bad = copy.deepcopy(rec)
        bad["questions"] = [q for q in bad["questions"] if q["type"] != "camera"]
        bad["questions"] = [{**q, "id": f"q{i}", "depends_on": []} for i, q in enumerate(bad["questions"])]
        self.assertTrue(any("camera" in e for e in rv2.validate_record(bad, p)))
        orbit = prompts[42]  # "... while the camera flies rapidly around it in a broad continuous orbit ..."
        cam = {"id": "q0", "type": "camera", "question": "Does the camera stay locked?", "depends_on": [],
               "evidence": "the camera flies rapidly around it", "region": "background", "expect": "still"}
        inverted = {k: orbit[k] for k in ("prompt_id", "family_id", "motion_level", "detail_level")}
        inverted["questions"] = [cam]
        self.assertTrue(any("camera moving" in e for e in rv2.validate_record(inverted, orbit)))
        inverted["questions"] = [{**cam, "question": "Does the camera orbit the lighthouse?", "expect": "moving"}]
        self.assertFalse(any("camera" in e for e in rv2.validate_record(inverted, orbit)))
        rolls = {"id": "q0", "type": "spatial", "question": "Does the ball roll across the table?", "depends_on": [],
                 "evidence": orbit["prompt"].split()[0]}
        inverted["questions"] = [rolls]
        self.assertTrue(any("single frames" in e for e in rv2.validate_record(inverted, orbit)))
        inverted["questions"] = [{**rolls, "question": "Is the spinning top on the table?"}]
        self.assertFalse(any("single frames" in e for e in rv2.validate_record(inverted, orbit)))
        test_prompt = next(q for q in prompts.values() if q["split"] == "test")
        self.assertTrue(any("locked test" in e for e in rv2.validate_record({"prompt_id": test_prompt["prompt_id"]}, test_prompt)))


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.prompts = prompt_index()
        self.pilot = {r["prompt_id"]: r for r in rv2.load_jsonl(PILOT)}
        self.examples = [r for r in self.pilot.values() if r["family_id"] == "watchmaking"]

    def test_examples_are_turns_and_target_comes_last(self):
        msgs = rv2.build_messages(self.prompts[100], self.examples, self.prompts)
        self.assertEqual(len(msgs), 2 + 2 * len(self.examples))
        self.assertIn(self.prompts[100]["prompt"], msgs[-1]["content"])
        self.assertEqual(rv2.parse_reply(msgs[2]["content"])["questions"][0]["id"], "q0")

    def test_errors_are_fed_back_until_valid(self):
        good = json.dumps({"questions": self.pilot[100]["questions"]})
        broken = copy.deepcopy(self.pilot[100]["questions"])
        broken[0]["evidence"] = "not in the prompt"
        replies = iter(["Sure! Here it is:", "```json\n" + json.dumps({"questions": broken}) + "\n```", good])
        seen = []

        def chat(msgs):
            seen.append(msgs)
            return next(replies)

        result = rv2.generate_one(self.prompts[100], self.examples, self.prompts, chat, repairs=2)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertIn("evidence", seen[2][-1]["content"])  # the second error went back to the model
        self.assertEqual(result["record"]["motion_level"], self.prompts[100]["motion_level"])

        replies = iter(["no json"] * 3)
        self.assertFalse(rv2.generate_one(self.prompts[100], self.examples, self.prompts,
                                          lambda m: next(replies), repairs=2)["ok"])

    def test_evidence_is_snapped_to_prompt_casing(self):
        text = "One oversized king stands still.  The camera is locked."
        qs = rv2.snap_evidence([{"evidence": "the camera is locked"}, {"evidence": "stands  still"},
                                {"evidence": "one oversized king"}, {"evidence": "a queen"}], text)
        self.assertEqual([q["evidence"] for q in qs],
                         ["The camera is locked", "stands still", "One oversized king", "a queen"])

    def test_repair_message_says_how_to_fix_evidence(self):
        p = self.prompts[100]  # "... there are no patterns or other objects, ..."
        def msg(ev):
            return rv2.repair_message([], {"questions": [{"id": "q6", "evidence": ev}]}, p)
        self.assertIn("'other objects'", msg("no other objects"))
        self.assertIn("remove the question", msg("From a stationary overhead view"))
        self.assertIn("without '...'", msg("plain white paper crane... light gray table"))
        self.assertIn("split it", msg("sits motionless on an empty table. The locked camera frames it"))

    def test_edits_drop_renumber_and_rewrite_dependencies(self):
        rec = {"prompt_id": 7, "questions": [
            {"id": "q0", "type": "entity", "depends_on": []},
            {"id": "q1", "type": "negative", "depends_on": ["q0"]},
            {"id": "q2", "type": "attribute", "depends_on": ["q0", "q1"]}]}
        out = rv2.apply_edits(rec, [{"id": "q1", "action": "drop"},
                                    {"id": "q2", "action": "set", "fields": {"type": "detail"}}])
        self.assertEqual([(q["id"], q["type"], q["depends_on"]) for q in out["questions"]],
                         [("q0", "entity", []), ("q1", "detail", ["q0"])])
        self.assertEqual(rec["questions"][2]["type"], "attribute")  # the draft is not modified
        with self.assertRaises(ValueError):
            rv2.apply_edits(rec, [{"id": "q9", "action": "drop"}])

    def test_reference_against_itself_has_full_recall(self):
        import contextlib, io, tempfile

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "r.md"
            args = argparse.Namespace(generated=str(PILOT), reference=str(PILOT), prompts=[str(p) for p in PROMPTS],
                                      exclude_family=["watchmaking"], report=str(report))
            with contextlib.redirect_stdout(io.StringIO()):
                rv2.compare(args)
            text = report.read_text(encoding="utf-8")
        self.assertIn("= 100.0%", text)
        self.assertIn("matching no reference question: 0", text)


if __name__ == "__main__":
    unittest.main()
