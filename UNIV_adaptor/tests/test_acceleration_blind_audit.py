"""Tests for blinding, score-independent sampling, lineage and missing metrics."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import contextlib
import io
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from UNIV_adaptor.scripts.data import acceleration_blind_audit as audit


def fixture():
    rows = []
    for prompt in range(4):
        for seed in range(3):
            for action in ("FULL", "S", "T"):
                key = f"p{prompt}_s{seed}_{action}"
                rows.append({"source": "testmodel", "id": key, "group": f"{prompt}_{seed}",
                             "action": action, "prompt": f"Prompt {prompt}", "prompt_key": audit.digest(str(prompt)),
                             "seed": str(seed), "path": f"/videos/{key}.mp4", "sha256": audit.digest(key),
                             "cell": str(prompt % 2), "family": str(prompt), "scores": {"vbench5": .8 if action == "FULL" else .7}})
    config = {"seed": 123, "minimum_raters": 3, "metric_epsilon": {"vbench5": .001},
              "presentation": {"width": 1280, "height": 720, "fps": 24, "crf": 16},
              "real_strata": [{"source": "testmodel", "left": "FULL", "right": "S", "count": 4}],
              "synthetic": {"source": "testmodel", "action": "FULL", "bases": 2},
              "seed_controls": {"source": "testmodel", "action": "FULL", "count": 2}}
    return config, {"testmodel": rows}


class BlindAuditTests(unittest.TestCase):
    def test_sampling_ignores_scores_and_is_repeatable(self):
        config, sources = fixture()
        expected = audit.make_plan(config, sources)
        changed = copy.deepcopy(sources)
        for i, row in enumerate(changed["testmodel"]):
            row["scores"] = {"vbench5": i * 1000}
        self.assertEqual([p["id"] for p in expected], [p["id"] for p in audit.make_plan(config, changed)])
        self.assertEqual(expected, audit.make_plan(config, sources))
        self.assertEqual(len(expected), 18)
        self.assertEqual(len({p["cluster"] for p in expected if p["kind"] == "synthetic"}), 2)
        for p in expected:
            if p["kind"] == "real":
                self.assertEqual(p["a"]["seed"], p["b"]["seed"])
            if p["kind"] == "seed_control":
                self.assertNotEqual(p["a"]["seed"], p["b"]["seed"])

    def test_no_silent_quota_reduction(self):
        config, sources = fixture()
        config["real_strata"][0]["count"] = 100
        with self.assertRaises(ValueError):
            audit.make_plan(config, sources)

    def test_session_resume_sides_and_private_fields(self):
        plan = {"plan_sha256": "frozen"}
        package = {"pairs": [{"id": str(i), "a": "left", "b": "right", "prompt": "prompt"} for i in range(20)]}
        first = audit.session(plan, package, "rater01")
        self.assertEqual(first, audit.session(plan, package, "rater01"))
        self.assertNotEqual(first, audit.session(plan, package, "rater02"))
        for row in first:
            self.assertEqual(row["A"], "right" if row["swap"] else "left")
            self.assertEqual(set(row), {"id", "prompt", "A", "B", "swap"})
        with self.assertRaises(ValueError):
            audit.session(plan, package, "../private")

    def test_report_unblinds_votes_and_does_not_copy_synthetic_scores(self):
        config, sources = fixture()
        pairs = audit.make_plan(config, sources)
        plan = {"config": config, "pairs": pairs}
        plan["plan_sha256"] = audit.digest(plan)
        package = {"plan_sha256": plan["plan_sha256"], "clips": {}, "pairs": [
            {"id": p["id"], "a": "left", "b": "right", "prompt": p["a"]["prompt"]} for p in pairs]}
        package["package_sha256"] = audit.digest(package)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            audit.write(out / "private/plan.json", plan)
            audit.write(out / "private/package.json", package)
            for i in range(3):
                participant = f"rater{i}"
                mapping = audit.session(plan, package, participant)
                # Every rater prefers canonical first video, despite side randomization.
                answers = {p["id"]: {d: "B" if p["swap"] else "A" for d in audit.DIMENSIONS} for p in mapping}
                audit.write(out / f"private/ratings/{participant}.json", {"participant": participant,
                    "plan_sha256": plan["plan_sha256"], "package_sha256": package["package_sha256"], "answers": answers})
            with contextlib.redirect_stdout(io.StringIO()):
                audit.report(SimpleNamespace(out=out, presented_scores=None))
            result = audit.read(out / "analysis/report.json")
            self.assertEqual(result["fully_rated_pairs"], 18)
            self.assertTrue(result["metric_summary"])
            self.assertTrue(all(r["kind"] != "synthetic" for r in result["metric_summary"]))
            self.assertTrue(all(r["pair_accuracy"] == 1 for r in result["metric_summary"] if r["kind"] == "real"))
            bad = out / "bad.csv"
            bad.write_text("clip_id,video_sha256,vbench5\nunknown,bad,.9\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit.report(SimpleNamespace(out=out, presented_scores=bad))
            # Fewer than three raters must not produce a metric-validation result,
            # and an old nonempty metric CSV must not survive the new report.
            third = out / "private/ratings/rater2.json"
            data = audit.read(third)
            data["answers"] = {}
            audit.write(third, data)
            with contextlib.redirect_stdout(io.StringIO()):
                audit.report(SimpleNamespace(out=out, presented_scores=None))
            result = audit.read(out / "analysis/report.json")
            self.assertEqual(result["fully_rated_pairs"], 0)
            self.assertEqual(result["metric_summary"], [])
            self.assertEqual(len((out / "analysis/metric_pairs.csv").read_text(encoding="utf-8-sig").splitlines()), 1)

    def test_filter_keeps_spatial_distortion_on_native_grid(self):
        config, _ = fixture()
        f = audit.render_filter(config["presentation"], {"kind": "spatial", "level": .5},
                                {"width": 832, "height": 480, "duration": 5})
        self.assertIn("scale=416:240:flags=area,scale=832:480:flags=lanczos", f)
        with self.assertRaises(ValueError):
            audit.render_filter(config["presentation"], None, {"width": 1920, "height": 1080, "duration": 5})

    def test_plan_identity_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            body = {"pairs": []}
            body["plan_sha256"] = audit.digest(body)
            audit.immutable(out / "private/plan.json", body)
            audit.immutable(out / "private/plan.json", body)
            self.assertEqual(audit.load_plan(out), body)
            body["pairs"] = ["tamper"]
            with self.assertRaises(ValueError):
                audit.immutable(out / "private/plan.json", body)
            audit.write(out / "private/plan.json", body)
            with self.assertRaises(ValueError):
                audit.load_plan(out)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg integration test requires executables in PATH")
    def test_real_encoding_http_blinding_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            config, sources = fixture()
            config["presentation"].update(width=320, height=240)
            config["real_strata"][0]["count"] = 1
            config["synthetic"]["bases"] = 1
            config["seed_controls"]["count"] = 1
            rows = sources["testmodel"][:6]  # one prompt, two seeds, three actions
            for i, row in enumerate(rows):
                path = out / f"source{i}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                                "testsrc2=size=64x48:rate=24:duration=3", "-vf", f"hue=h={i * 20}",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
                row.update(path=str(path), sha256=audit.file_hash(path))
            body = {"config": config, "pairs": audit.make_plan(config, {"testmodel": rows})}
            body["plan_sha256"] = audit.digest(body)
            audit.write(out / "private/plan.json", body)
            with contextlib.redirect_stdout(io.StringIO()):
                audit.package(SimpleNamespace(out=out, path_map=[]))
                first = audit.read(out / "private/package.json")
                audit.package(SimpleNamespace(out=out, path_map=[]))
            self.assertEqual(first, audit.read(out / "private/package.json"))
            self.assertEqual(len(first["pairs"]), 8)
            for clip in first["clips"].values():
                self.assertEqual(clip["info"]["width"], 320)
                self.assertAlmostEqual(clip["info"]["duration"], 3, delta=.15)
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            process = subprocess.Popen([sys.executable, audit.__file__, "serve", "--out", str(out), "--port", str(port)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            base = f"http://127.0.0.1:{port}"
            try:
                for _ in range(50):
                    try:
                        with urllib.request.urlopen(base, timeout=1) as response:
                            self.assertIn("视频配对评价", response.read().decode())
                        break
                    except urllib.error.URLError:
                        time.sleep(.1)
                else:
                    self.fail("HTTP server did not start")
                def post(route, payload):
                    request = urllib.request.Request(base + route, data=json.dumps(payload).encode(),
                                                     headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request) as response:
                        return json.load(response)
                data = post("/api/session", {"participant": "rater01"})
                p = data["pairs"][0]
                self.assertEqual(set(p), {"id", "prompt", "A", "B"})
                for route in ("/private/plan.json", "/private/package.json", "/analysis/report.json"):
                    with self.assertRaises(urllib.error.HTTPError) as exc:
                        urllib.request.urlopen(base + route)
                    self.assertEqual(exc.exception.code, 404)
                    exc.exception.close()
                votes = {d: "tie" for d in audit.DIMENSIONS}
                post("/api/rate", {"participant": "rater01", "pair": p["id"], "votes": votes})
                self.assertEqual(post("/api/session", {"participant": "rater01"})["answers"][p["id"]], votes)
                self.assertFalse(post("/api/session", {"participant": "rater02"})["answers"])
                request = urllib.request.Request(base + f"/media/{p['A']}.mp4", headers={"Range": "bytes=0-9"})
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(len(response.read()), 10)
            finally:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
