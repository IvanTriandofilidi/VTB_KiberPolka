import numpy as np
import pandas as pd
import pytest
import torch

from cybershelf.data import align, load_features, load_targets, submission
from cybershelf.features import FeatureBuilder
from cybershelf.metrics import macro_auc
from cybershelf.stacking import StackingNet, batch_indices, focal_loss, predict_logits


@pytest.fixture
def frame():
    return pd.DataFrame({"customer_id": [3, 1, 2], "cat_feature_1": [1., np.nan, 2.],
                         "num_feature_1": [np.nan, 2., 4.]})


def test_alignment_uses_ids_not_row_order(frame):
    ordered = align([1, 2, 3], frame)
    assert ordered.customer_id.tolist() == [1, 2, 3]
    assert ordered.num_feature_1.iloc[0] == 2


def test_duplicate_and_missing_ids_fail(frame):
    with pytest.raises(ValueError, match="unique"):
        align([3, 1, 2], pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError, match="sets differ"):
        align([3, 1, 9], frame)


def test_extra_join_and_target_alignment(tmp_path, frame):
    main = tmp_path / "main.parquet"
    extra = tmp_path / "extra.parquet"
    target = tmp_path / "targets.parquet"
    frame.to_parquet(main, index=False)
    pd.DataFrame({"customer_id": [2, 3, 1], "num_feature_2": [20, 30, 10]}).to_parquet(extra)
    pd.DataFrame({"customer_id": [1, 2, 3], "target_1_1": [0, 1, 0]}).to_parquet(target)
    x = load_features(main, extra)
    assert x.num_feature_2.tolist() == [30, 10, 20]
    assert load_targets(target, x.customer_id, expected_count=1).iloc[:, 0].tolist() == [0, 0, 1]


def test_target_nonbinary_rejected(tmp_path):
    p = tmp_path / "targets.parquet"
    pd.DataFrame({"customer_id": [1, 2], "target_1_1": [0, 2]}).to_parquet(p)
    with pytest.raises(ValueError, match="binary"):
        load_targets(p, [1, 2], expected_count=1)


def test_transform_uses_fitted_statistics_and_unknown_categories(frame):
    builder = FeatureBuilder().fit(frame)
    unseen = frame.iloc[:1].copy()
    unseen["cat_feature_1"] = 999
    xgb = builder.transform(unseen, "xgboost")
    assert xgb.cat_feature_1.isna().all()
    cb = builder.transform(unseen, "catboost")
    assert cb.cat_feature_1.iloc[0] == "999"
    assert cb.num_feature_missing_count.iloc[0] == pytest.approx(
        (1 - builder.count_mean) / builder.count_std
    )
    assert pd.isna(frame.cat_feature_1.iloc[1])


def test_schema_and_fractional_category_fail(frame):
    builder = FeatureBuilder().fit(frame)
    with pytest.raises(ValueError, match="schema"):
        builder.transform(frame.drop(columns="num_feature_1"), "catboost")
    frame.loc[0, "cat_feature_1"] = 1.5
    with pytest.raises(ValueError, match="integral"):
        FeatureBuilder().fit(frame)


def test_undefined_auc_is_not_replaced_with_point_five():
    result = macro_auc([[0, 0], [1, 0]], [[.1, .1], [.9, .8]], ["a", "b"])
    assert result["macro_auc"] is None
    assert result["defined_target_mean_auc"] == 1
    assert result["undefined_targets"] == ["b"]


def test_submission_order_shape_and_nonfinite():
    result = submission([12, 5], np.zeros((2, 2)), ["target_2_1", "target_1_1"])
    assert result.columns.tolist() == ["customer_id", "predict_2_1", "predict_1_1"]
    assert result.customer_id.tolist() == [12, 5]
    with pytest.raises(ValueError):
        submission([12, 5], [[float("inf"), 0], [0, 0]], ["target_a", "target_b"])


@pytest.mark.parametrize("length,batch", [(3, 2), (9, 4), (513, 512), (10, 3)])
def test_batchnorm_batches_keep_every_row(length, batch):
    parts = batch_indices(length, batch, np.random.default_rng(42))
    assert min(map(len, parts)) >= 2
    assert sorted(np.concatenate(parts).tolist()) == list(range(length))


def test_focal_gradient_finite_at_extreme_logits():
    logits = torch.tensor([[-1000., 1000.], [3., -3.]], requires_grad=True)
    loss = focal_loss(logits, torch.tensor([[1., 0.], [1., 0.]]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_stacker_shape_eval_and_tail():
    torch.set_num_threads(2)
    model = StackingNet()
    scores = predict_logits(model, np.zeros((7, 82), dtype="float32"), batch_size=3)
    assert scores.shape == (7, 41)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores[0], scores[-1], atol=1e-6)
