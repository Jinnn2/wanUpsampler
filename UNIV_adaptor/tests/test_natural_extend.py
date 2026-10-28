import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from UNIV_adaptor.scripts.data import natural_extend as ne  # noqa: E402
from UNIV_adaptor.scripts.data import natural_prompts as npr  # noqa: E402


class FakeChat:
    def __init__(self):
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return f'"{messages[-1]["content"]} rendered in rich cinematic detail as the camera slowly pans"'


class FakeJudge:
    def p_yes(self, prompts, images=None, videos=None, metadata=None):
        return [0.9 if "pans" in p[-1] and "substantial motion" in p[-1] else 0.1 for p in prompts]


class ExtendTest(unittest.TestCase):
    def test_extend_keeps_family_offsets_id_and_resumes(self):
        shorts = [json.loads(line) for line in (REPO / "prompts/univ_natural_v1.jsonl").read_text(encoding="utf-8").splitlines()[:5]]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "short.jsonl"
            src.write_text("".join(json.dumps(s) + "\n" for s in shorts), encoding="utf-8")
            args = type("A", (), {"prompts": str(src), "model": "fake", "out": tmp, "prompts_out": None, "batch": 2})
            chat = FakeChat()
            ne.extend(args, chat=chat, judge=FakeJudge())
            self.assertEqual(len(chat.calls), 5)
            self.assertEqual(chat.calls[0][0], {"role": "system", "content": ne.LM_EN_SYS_PROMPT})
            recs = [json.loads(line) for line in (Path(tmp) / "univ_natural_ext_v1.jsonl").read_text(encoding="utf-8").splitlines()]
            for s, r in zip(shorts, recs):
                self.assertEqual(r["prompt_id"], s["prompt_id"] + 1000)
                self.assertEqual(r["family_id"], s["family_id"])
                self.assertEqual(r["short_prompt"], s["prompt"])
                self.assertFalse(r["prompt"].startswith('"'))
                self.assertEqual((r["split"], r["motion_level"], r["detail_level"]), ("natural_ext", "high", "low"))
            again = FakeChat()
            ne.extend(args, chat=again, judge=FakeJudge())  # every prompt cached: no new generation
            self.assertEqual(again.calls, [])

    def test_build_configs_defaults_reproduce_natural_v1(self):
        prompts = [json.loads(line) for line in (REPO / "prompts/univ_natural_v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        with tempfile.TemporaryDirectory() as tmp:
            npr.build_configs(prompts, tmp)
            for name in ("platform", "twostage"):
                got = json.loads((Path(tmp) / f"published_wan21_{name}_natural.json").read_text(encoding="utf-8"))
                want = json.loads((REPO / f"UNIV_adaptor/configs/published_wan21_{name}_natural.json").read_text(encoding="utf-8"))
                self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
