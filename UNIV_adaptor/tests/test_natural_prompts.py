import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from UNIV_adaptor.scripts.data import natural_prompts as nat  # noqa: E402

CLEAN = {k: "0.001" for k in nat.NSFW}


class FilterTest(unittest.TestCase):
    def test_clean_strips_pika_flags(self):
        self.assertEqual(nat.clean("a red fox runs through snow at dawn -gs 12 -motion 2 -ar 16:9"),
                         "a red fox runs through snow at dawn")
        self.assertEqual(nat.clean("a sign saying &quot;hi&quot; &amp; a fox"), 'a sign saying "hi" & a fox')

    def test_keep(self):
        ok = "a red fox runs through fresh snow at dawn"
        self.assertTrue(nat.keep(ok, CLEAN))
        self.assertFalse(nat.keep("a fox", CLEAN))  # too short
        self.assertFalse(nat.keep(ok + " see https://x.y", CLEAN))
        self.assertFalse(nat.keep("a naked man runs through fresh snow at dawn", CLEAN))
        self.assertFalse(nat.keep(ok, {**CLEAN, "obscene": "0.3"}))
        self.assertFalse(nat.keep("一只狐狸 在 雪地 里 奔跑 的 样子 很 美", CLEAN))


if __name__ == "__main__":
    unittest.main()
