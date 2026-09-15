import numpy as np
import pytest

from nnopt.cascade.error_propagation import (
    InfluenceModel,
    fit_aggregation,
    influence_coefficients,
    relative_prediction_error,
)


def test_coefficients_are_global_over_local():
    c = influence_coefficients({"a": 0.20, "b": 0.05}, {"a": 0.10, "b": 0.10})
    assert c["a"] == pytest.approx(2.0)
    assert c["b"] == pytest.approx(0.5)


def test_zero_local_error_does_not_divide_by_zero():
    c = influence_coefficients({"a": 0.1}, {"a": 0.0})
    assert np.isfinite(c["a"])


def test_linear_aggregation_sums_weighted_terms():
    m = InfluenceModel({"a": 2.0, "b": 1.0}, p=1.0, scale=1.0)
    assert m.predict({"a": 0.1, "b": 0.2}) == pytest.approx(0.4)


def test_quadratic_aggregation_is_root_sum_of_squares():
    m = InfluenceModel({"a": 1.0, "b": 1.0}, p=2.0, scale=1.0)
    assert m.predict({"a": 0.3, "b": 0.4}) == pytest.approx(0.5)


def test_weighted_errors_are_what_the_allocator_should_minimize():
    m = InfluenceModel({"a": 3.0, "b": 0.5})
    w = m.weighted_errors({"a": 0.1, "b": 0.1})
    assert w["a"] > w["b"], "an operator with 6x the influence must dominate"


def test_operator_absent_from_coefficients_contributes_nothing():
    m = InfluenceModel({"a": 1.0}, p=1.0)
    assert m.predict({"a": 0.1, "unknown": 5.0}) == pytest.approx(0.1)


def test_empty_config_predicts_zero():
    assert InfluenceModel({"a": 1.0}).predict({}) == 0.0


def test_fit_recovers_a_linear_generating_rule():
    coeffs = {"a": 2.0, "b": 1.0, "c": 0.5}
    truth = InfluenceModel(coeffs, p=1.0, scale=1.0)
    configs = []
    rng = np.random.default_rng(0)
    for _ in range(6):
        local = {k: float(rng.uniform(0.01, 0.2)) for k in coeffs}
        configs.append((local, truth.predict(local)))
    fitted = fit_aggregation(coeffs, configs)
    assert fitted.p == pytest.approx(1.0)
    assert max(relative_prediction_error(fitted, configs)) < 1e-6


def test_fit_recovers_a_quadratic_generating_rule():
    coeffs = {"a": 1.0, "b": 1.0, "c": 1.0}
    truth = InfluenceModel(coeffs, p=2.0, scale=1.0)
    configs = []
    rng = np.random.default_rng(1)
    for _ in range(6):
        local = {k: float(rng.uniform(0.01, 0.2)) for k in coeffs}
        configs.append((local, truth.predict(local)))
    fitted = fit_aggregation(coeffs, configs)
    assert fitted.p == pytest.approx(2.0)
    assert max(relative_prediction_error(fitted, configs)) < 1e-6


def test_fit_with_no_configs_is_a_safe_identity():
    m = fit_aggregation({"a": 1.0}, [])
    assert m.p == 1.0 and m.scale == 1.0


def test_fit_scales_when_only_one_config_available():
    coeffs = {"a": 1.0}
    configs = [({"a": 0.1}, 0.5)]     # measured is 5x the raw sum
    m = fit_aggregation(coeffs, configs)
    assert m.predict({"a": 0.1}) == pytest.approx(0.5, rel=1e-6)
