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
        test_prompt = next(q for q in prompts.values() if q["split"] == "test")
        self.assertTrue(any("locked test" in e for e in rv2.validate_record({"prompt_id": test_prompt["prompt_id"]}, test_prompt)))


if __name__ == "__main__":
    unittest.main()
