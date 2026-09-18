import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from UNIV_adaptor.data_protocol import canonical_sha256, write_json_atomic
from UNIV_adaptor.scripts.data.score_phase2_dataset import collect, SCORE_SCHEMA
from UNIV_adaptor.scripts.data.phase2_analysis import DIMENSIONS
from UNIV_adaptor.tests.test_phase2_quality import fixture, synthetic_scores
from UNIV_adaptor.scripts.router.phase2_gain_model import (
    cross_validate, fit_ridge, fit_text, predict, select_actions, text_features, train_mixture,
)
from UNIV_adaptor.scripts.router.train_phase2_gain_prior import (
    ACTION_SETS, calibrated_costs, evaluate, load_data, train, load_t5,
)


class GainModelTests(unittest.TestCase):
    def test_fold_local_vocabulary_and_unseen_text(self):
        state = fit_text(["red running fox", "blue running fish"])
        self.assertNotIn("validationonly",state["vocabulary"])
        x = text_features(["validationonly", "red fox"],state)
        self.assertTrue(np.all(x[0]==0))
        self.assertAlmostEqual(np.linalg.norm(x[1]),1.)

    def test_learns_gain_signal_without_winner_labels(self):
        texts = ["red motion" if i%2 else "blue static" for i in range(40)]
        y = np.array([[.1 if i%2 else -.1] for i in range(40)])
        model,state,oof,folds,cv = cross_validate(texts,y,folds=5)
        self.assertIsNotNone(next(r["alpha"] for r in cv if r["selected"]))
        self.assertLess(np.mean((oof-y)**2),.001)
        prediction = predict(text_features(["red motion","blue static"],state),model)
        self.assertGreater(prediction[0,0],.08)
        self.assertLess(prediction[1,0],-.08)
        self.assertEqual(set(folds),set(range(5)))

    def test_constant_labels_choose_mean_control(self):
        y = np.ones((20,2))*.04
        model,state,oof,folds,cv = cross_validate([f"prompt {i}" for i in range(20)],y)
        self.assertIsNone(next(r["alpha"] for r in cv if r["selected"]))
        np.testing.assert_allclose(oof,y)

    def test_budget_choice_uses_only_predictions_and_calibration(self):
        p = np.array([[0,.1,.2],[0,-.1,.9]])
        chosen, eligible = select_actions(p,[10,20,30],20)
        np.testing.assert_array_equal(chosen,[1,0])
        chosen,_ = select_actions(p,[10,20,30],9)
        self.assertIsNone(chosen)

    def test_train_mixture_frontier(self):
        weights = train_mixture(np.array([.5,.4,.9]),np.array([10,20,30]),np.arange(3),20)
        np.testing.assert_allclose(weights,[.5,0,.5])
        self.assertAlmostEqual(float(weights@np.array([10,20,30])),20)

    def test_mean_model_and_serialized_weights(self):
        x = np.eye(4);y=np.array([[0],[1],[2],[3]])
        model=fit_ridge(x,y,None)
        np.testing.assert_allclose(predict(x,model),np.ones((4,1))*1.5)


class GainPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        fixture(self.root)
        rows,identity=collect(self.root)
        digest=canonical_sha256({"identity":identity,"rows":rows})
        write_json_atomic(self.root/"evaluation_inputs.json",dict(input_sha256=digest,**identity,rows=rows))
        body=dict(schema=SCORE_SCHEMA,input_sha256=digest,dimensions=list(DIMENSIONS),scores=synthetic_scores(rows),provenance={})
        write_json_atomic(self.root/"scores.json",dict(**body,payload_sha256=canonical_sha256(body)))

    def args(self,out):
        return argparse.Namespace(action_set="three",out_dir=str(out),features="tfidf",t5_dir=None,
                                  folds=2,seed=10,alphas=[.1,1.],max_features=100,budgets_seconds=None)

    def test_offline_loader_checks_hash_and_averages_seeds(self):
        samples,rows,identity=load_data(self.root,ACTION_SETS["three"])
        self.assertEqual(len(samples),4)
        self.assertEqual(len(samples[-1]["seed_seconds"][0]),3)
        scores=json.loads((self.root/"scores.json").read_text())
        scores["scores"][next(iter(scores["scores"]))][DIMENSIONS[0]]=.123
        write_json_atomic(self.root/"scores.json",scores)
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            load_data(self.root,ACTION_SETS["three"])

    def test_validation_labels_and_latency_do_not_change_trained_model(self):
        samples,rows,identity=load_data(self.root,ACTION_SETS["three"])
        train(self.args(self.root/"first"),samples,rows,identity)
        changed=copy.deepcopy(samples);changed_rows=copy.deepcopy(rows)
        for s in changed:
            if s["split"]=="validation":
                s["quality"]=[.9,.1,.2];s["seconds"]=[999.,999.,999.]
                s["seed_seconds"]=[[999.]*3 for _ in range(3)]
        for r in changed_rows:
            if r["split"]=="validation":r["pipeline_seconds"]=999.
        train(self.args(self.root/"second"),changed,changed_rows,identity)
        with np.load(self.root/"first/model.npz") as a,np.load(self.root/"second/model.npz") as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        first=json.loads((self.root/"first/model.json").read_text())
        second=json.loads((self.root/"second/model.json").read_text())
        self.assertEqual(first["train_p95_seconds"],second["train_p95_seconds"])
        self.assertEqual(first["selected_alpha"],second["selected_alpha"])
        result=json.loads((self.root/"second/policy_summary.json").read_text())
        self.assertTrue(all(r["video_cap_violation_rate"]==1 for r in result if r["split"]=="validation_development"))

    def test_permutation_baseline_matches_histogram_cost_and_exposes_matching(self):
        samples,rows,_=load_data(self.root,ACTION_SETS["three"])
        s=copy.deepcopy(samples[:2])
        s[0]["quality"]=[.8,.1,.1];s[1]["quality"]=[.1,.8,.1]
        pred=np.array([[0,-1,-1],[0,1,-1]])
        result,_=evaluate(s,pred,ACTION_SETS["three"],np.array([.5,.5,.4]),np.array([10,20,30]),
                          np.array([10,20,30]),[30],"test",42)
        by={r["policy"]:r for r in result}
        self.assertAlmostEqual(by["prompt_gain"]["calibrated_mean_seconds"],by["shuffled_router_hist_expected"]["calibrated_mean_seconds"])
        self.assertGreater(by["prompt_gain"]["gain_vs_histogram_shuffle"],.3)
        self.assertEqual(by["shuffled_router_hist_expected"]["action_fractions"],by["prompt_gain"]["action_fractions"])

    def test_budget_below_calibrated_min_is_explicit_error(self):
        samples,rows,identity=load_data(self.root,ACTION_SETS["three"])
        args=self.args(self.root/"invalid");args.budgets_seconds=[.01]
        with self.assertRaisesRegex(ValueError,"no eligible action"):
            train(args,samples,rows,identity)

    def test_t5_features_bind_to_prompt_text_and_hash(self):
        samples,_,_=load_data(self.root,ACTION_SETS["three"])
        emb=self.root/"t5";emb.mkdir()
        entries=[]
        import hashlib
        from UNIV_adaptor.data_protocol import sha256_file
        for i,s in enumerate(samples):
            path=emb/f"prompt_{i:06d}.npz"
            np.savez(path,pooled_embedding=np.ones(4096)*i)
            entries.append(dict(prompt_text=s["prompt"],prompt_sha256=hashlib.sha256(s["prompt"].encode()).hexdigest(),npz_file=str(path),npz_sha256=sha256_file(path)))
        body=dict(complete=True,backend="wan_native",prompts=entries)
        write_json_atomic(emb/"t5_manifest.json",dict(schema="prompt_t5_embeddings_manifest_v2",**body,manifest_sha256=canonical_sha256(body)))
        x,digest=load_t5(emb,samples)
        self.assertEqual(x.shape,(4,4096))
        (emb/"prompt_000000.npz").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError,"feature hash"):
            load_t5(emb,samples)


if __name__=="__main__":
    unittest.main()
