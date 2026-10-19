import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from UNIV_adaptor.scripts.data import twostage_wan21 as driver
from UNIV_adaptor.scripts.data import twostage_wan21_worker as worker


def config():
    return driver.read(driver.DEFAULT_CONFIG)


class ConfigTest(unittest.TestCase):
    def test_default_config_is_valid(self):
        cfg = driver.validate_config(config())
        self.assertEqual([a["id"] for a in cfg["arms"]], ["FULL50", "LR_FLASH", "LR_BICUBIC", "F41_RIFE", "F41_DUP"])

    def test_locked_test_prompt_is_rejected(self):
        cfg = config()
        cfg["prompts"][0]["split"] = "test"
        with self.assertRaises(ValueError):
            driver.validate_config(cfg)

    def test_spatial_arm_must_generate_half_canvas(self):
        cfg = config()
        cfg["arms"][1]["generation"] = "HR41"
        with self.assertRaises(ValueError):
            driver.validate_config(cfg)

    def test_temporal_arm_must_generate_half_length(self):
        cfg = config()
        cfg["generations"][2]["frame_num"] = 45
        with self.assertRaises(ValueError):
            driver.validate_config(cfg)

    def test_v2b_and_dev2_configs_are_valid(self):
        for name in ("published_wan21_twostage_v2b.json", "published_wan21_twostage_dev2.json"):
            cfg = driver.validate_config(driver.read(driver.DEFAULT_CONFIG.with_name(name)))
            self.assertEqual(cfg["arms"][0]["generation"], driver.EXTERNAL)
            arm = next(a for a in cfg["arms"] if a["id"] == "LR41_FLASH_RIFE")
            self.assertEqual(driver.chain_shapes(cfg, arm), [(41, 416, 240), (41, 832, 480), (81, 832, 480)])
        dev2 = driver.read(driver.DEFAULT_CONFIG.with_name("published_wan21_twostage_dev2.json"))
        v1 = config()
        self.assertFalse({p["prompt"] for p in dev2["prompts"]} & {p["prompt"] for p in v1["prompts"]})
        self.assertFalse({p["family_id"] for p in dev2["prompts"]} & {p["family_id"] for p in v1["prompts"]})

    def test_incomplete_chain_and_bad_rcm_steps_are_rejected(self):
        cfg = driver.read(driver.DEFAULT_CONFIG.with_name("published_wan21_twostage_v2b.json"))
        bad = copy.deepcopy(cfg)
        next(a for a in bad["arms"] if a["id"] == "LR41_FLASH_RIFE")["post"] = "flashvsr"
        with self.assertRaises(ValueError):
            driver.validate_config(bad)
        bad = copy.deepcopy(cfg)
        next(g for g in bad["generations"] if g["id"] == "RCM4")["steps"] = 5
        with self.assertRaises(ValueError):
            driver.validate_config(bad)
        bad = copy.deepcopy(cfg)
        bad["arms"][1]["generation"] = driver.EXTERNAL
        with self.assertRaises(ValueError):
            driver.validate_config(bad)


class JobTest(unittest.TestCase):
    def test_jobs_share_stage1_generations(self):
        cfg = config()
        gens, finals = driver.build_jobs(cfg, 8)
        groups = len(cfg["prompts"]) * len(cfg["seeds"])
        self.assertEqual(len(gens), 3 * groups)
        self.assertEqual(len(finals), 5 * groups)
        ids = {g["id"] for g in gens}
        self.assertTrue(all(f["generation_job"] in ids for f in finals))
        by_group = {}
        for f in finals:
            by_group.setdefault(f["group_id"], {})[f["arm"]["id"]] = f["generation_job"]
        for arms in by_group.values():
            self.assertEqual(arms["LR_FLASH"], arms["LR_BICUBIC"])
            self.assertEqual(arms["F41_RIFE"], arms["F41_DUP"])
        self.assertEqual({sum(g["gpu"] == i for g in gens) for i in range(8)}, {len(gens) // 8})


class ProcessorTest(unittest.TestCase):
    def test_duplicate_hold_gives_81_frames(self):
        import numpy as np
        frames = np.arange(41, dtype=np.uint8)[:, None, None, None].repeat(2, 1).repeat(2, 2).repeat(3, 3)
        output, _ = worker.duplicate_processor(None, None)(frames)
        self.assertEqual(output.shape[0], 81)
        self.assertEqual(list(output[:5, 0, 0, 0]), [0, 0, 1, 1, 2])
        self.assertEqual(output[-1, 0, 0, 0], 40)

    def test_bicubic_doubles_canvas(self):
        import numpy as np
        frames = np.zeros((3, 240, 416, 3), dtype=np.uint8)
        output, _ = worker.bicubic_processor(None, (832, 480))(frames)
        self.assertEqual(output.shape, (3, 480, 832, 3))

    def test_register_size_extends_wan_tables_in_place(self):
        configs = types.ModuleType("wan.configs")
        configs.SIZE_CONFIGS, configs.MAX_AREA_CONFIGS = {"832*480": (832, 480)}, {"832*480": 832 * 480}
        configs.SUPPORTED_SIZES = {"t2v-1.3B": ("480*832", "832*480")}
        package = types.ModuleType("wan")
        package.configs = configs
        saved = {k: sys.modules.get(k) for k in ("wan", "wan.configs")}
        sys.modules.update({"wan": package, "wan.configs": configs})
        try:
            sizes = configs.SIZE_CONFIGS
            self.assertTrue(worker.register_size("416*240"))
            self.assertIs(configs.SIZE_CONFIGS, sizes)
            self.assertEqual(sizes["416*240"], (416, 240))
            self.assertIn("416*240", configs.SUPPORTED_SIZES["t2v-1.3B"])
            self.assertEqual(worker.register_size("832*480"), [])
        finally:
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value


class FinalRecordTest(unittest.TestCase):
    def test_full50_uses_stage1_video_and_post_arms_need_receipts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        cfg = copy.deepcopy(config())
        cfg["prompts"], cfg["seeds"] = cfg["prompts"][:1], [42]
        gens, finals = driver.build_jobs(cfg, 1)
        p = {"plan_sha256": "abc", "gen_jobs": gens, "final_jobs": finals}
        for job in gens:
            video = out / (job["id"] + ".mp4")
            video.write_bytes(job["id"].encode())
            driver.write(driver.gen_record_path(out, job["id"]), {
                "plan_sha256": "abc", "job": job, "video_path": str(video), "video_sha256": driver.file_hash(video),
                "frames": None, "runtime": {"pipeline_seconds": 10.0}})
        args = types.SimpleNamespace(out=out)
        with self.assertRaises(RuntimeError):
            driver.final_records(args, p)
        for job in finals:
            if job["arm"]["post"] == "none":
                continue
            video = out / (job["id"] + ".mp4")
            video.write_bytes(job["id"].encode())
            driver.write(driver.final_record_path(out, job["id"]), {
                "plan_sha256": "abc", "job": job, "video_path": str(video), "video_sha256": driver.file_hash(video),
                "stage1_receipt_sha256": driver.file_hash(driver.gen_record_path(out, job["generation_job"])),
                "runtime": {"pipeline_seconds": 12.0, "stage1_seconds": 10.0, "post_seconds": 2.0}})
        records = driver.final_records(args, p)
        full = next(r for r in records if r["job"]["arm"]["id"] == "FULL50")
        self.assertEqual(full["runtime"]["post_seconds"], 0.0)
        self.assertTrue(full["video_path"].endswith("_HR81.mp4"))
        self.assertNotIn("gpu", full["job"])
        self.assertEqual(len(records), 5)
        json.dumps(records)

    def test_external_reference_and_post_chain(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out, ref = Path(tmp.name) / "run", Path(tmp.name) / "published_wan21_twostage_v1"
        cfg = driver.read(driver.DEFAULT_CONFIG.with_name("published_wan21_twostage_v2b.json"))
        cfg["prompts"], cfg["seeds"] = cfg["prompts"][:1], [42]
        cfg["arms"] = [a for a in cfg["arms"] if a["id"] in ("FULL50", "LR41_FLASH_RIFE")]
        gens, finals = driver.build_jobs(cfg, 1)
        gens = [g for g in gens if g["generation"]["id"] == "LR41"]
        source = ref / "full.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"full")
        driver.write(ref / "plan.json", {"plan_sha256": "ref", "schema": "x"})
        reference = driver.reference_binding(cfg, ref)
        p = {"plan_sha256": "abc", "gen_jobs": gens, "final_jobs": finals, "reference": reference}
        driver.write(ref / "dataset_manifest.json", {"plan_sha256": "ref", "partial_exploratory": False, "records": [{
            "job": {"id": "pilot_x_FULL50", "prompt": finals[0]["prompt"], "seed": 42, "arm": {"id": "FULL50"}},
            "video_path": str(source), "video_sha256": driver.file_hash(source),
            "runtime": {"pipeline_seconds": 100.0}}]})
        out.mkdir()
        gen = gens[0]
        video = out / "gen.mp4"
        video.write_bytes(b"gen")
        driver.write(driver.gen_record_path(out, gen["id"]), {
            "plan_sha256": "abc", "job": gen, "video_path": str(video), "video_sha256": driver.file_hash(video),
            "frames": None, "runtime": {"pipeline_seconds": 10.0}})
        chain = next(j for j in finals if j["arm"]["id"] == "LR41_FLASH_RIFE")
        self.assertFalse(driver.post_step_done(out, p, chain, 0))
        frames = out / "step0.npz"
        frames.write_bytes(b"frames")
        driver.write(driver.step_record_path(out, chain["id"], 0), {
            "plan_sha256": "abc", "job": chain, "step": 0, "seconds": 5.0,
            "input_receipt_sha256": driver.file_hash(driver.gen_record_path(out, gen["id"])),
            "frames": {"path": str(frames), "sha256": driver.file_hash(frames)}})
        self.assertTrue(driver.post_step_done(out, p, chain, 0))
        self.assertFalse(driver.post_step_done(out, p, chain, 1))
        final = out / "final.mp4"
        final.write_bytes(b"final")
        driver.write(driver.final_record_path(out, chain["id"]), {
            "plan_sha256": "abc", "job": chain, "video_path": str(final), "video_sha256": driver.file_hash(final),
            "stage1_receipt_sha256": driver.file_hash(driver.gen_record_path(out, gen["id"])),
            "step_receipt_sha256": [driver.file_hash(driver.step_record_path(out, chain["id"], 0))],
            "runtime": {"pipeline_seconds": 15.5, "stage1_seconds": 10.0, "post_seconds": 5.5}})
        records = {r["job"]["arm"]["id"]: r for r in driver.final_records(types.SimpleNamespace(out=out), p)}
        self.assertEqual(records["FULL50"]["video_path"], str(source))
        self.assertEqual(records["FULL50"]["runtime"]["pipeline_seconds"], 100.0)
        self.assertEqual(records["LR41_FLASH_RIFE"]["runtime"]["post_seconds"], 5.5)
        source.write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            driver.final_records(types.SimpleNamespace(out=out), p)


if __name__ == "__main__":
    unittest.main()
