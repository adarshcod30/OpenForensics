"""Every corruption must survive tf.function.

This is not hypothetical: `speckle` shipped with `tf.shape(x)[:2] + [1]`,
which adds 1 elementwise instead of appending a channel axis. It worked in
eager shape inference and failed only inside the graph -- i.e. only once a
multi-hour training run had already started.
"""
import numpy as np
import pytest
import tensorflow as tf

from openforensics.data.corruptions import CORRUPTIONS, random_corrupt

IMG = tf.constant(np.random.default_rng(0).random((64, 64, 3)), dtype=tf.float32)


@pytest.mark.parametrize("name,fn", CORRUPTIONS, ids=[n for n, _ in CORRUPTIONS])
def test_graph_safe_and_shape_preserving(name, fn):
    out = tf.function(fn)(IMG)
    assert tuple(out.shape) == (64, 64, 3)


@pytest.mark.parametrize("name,fn", CORRUPTIONS, ids=[n for n, _ in CORRUPTIONS])
def test_stays_in_unit_range(name, fn):
    out = tf.function(fn)(IMG).numpy()
    assert out.min() >= -1e-5 and out.max() <= 1 + 1e-5


def test_random_corrupt_is_stochastic_and_graph_safe():
    f = tf.function(lambda x: random_corrupt(x, prob=1.0, max_ops=2))
    means = {round(float(tf.reduce_mean(f(IMG))), 6) for _ in range(8)}
    assert len(means) > 1, "corruption draws are not varying"


def test_zero_probability_is_identity():
    out = tf.function(lambda x: random_corrupt(x, prob=0.0, max_ops=2))(IMG)
    np.testing.assert_allclose(out.numpy(), IMG.numpy())
