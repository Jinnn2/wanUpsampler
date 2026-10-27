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

    def test_tidy_drops_inline_parameters(self):
        self.assertEqual(nat.tidy("a tree in the wind, high detail -fps24 -motion0 -gs15"), "a tree in the wind, high detail")
        self.assertEqual(nat.tidy("windy day,-ar 9:16"), "windy day")
        self.assertEqual(nat.tidy("cinema 4d, --ar3:2"), "cinema 4d")
        self.assertEqual(nat.tidy("full of equipment,(ar-16:9)"), "full of equipment")
        self.assertEqual(nat.tidy("movie style,-camera pan right"), "movie style,camera pan right")
        self.assertEqual(nat.tidy("Dancing , slow motion 2x, a car"), "Dancing , slow motion 2x, a car")  # text kept

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
