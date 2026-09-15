"""Predicting end-to-end damage from per-operator error.

The gap this closes (README Sec 8.3.10-F): allocating rank to minimize the
plain SUM of per-operator errors reduced that sum by 12.8% but improved WER
by 58%. The sum is therefore not what the network's output actually
responds to -- some operators' errors are amplified on their way to the
output and others are absorbed, and a scheme that treats them as equal
misallocates capacity.

Model. Perturbing operator i by a relative output error E_i produces an
end-to-end error contribution c_i * E_i, where the influence coefficient

    c_i = E_glob(only operator i perturbed) / E_loc(operator i)

is measured ONCE per operator and then reused. For a configuration that
perturbs many operators, two aggregation rules bracket reality:

    linear (worst case, errors aligned):      E_glob ~ sum_i c_i * E_i
    quadratic (errors independent):           E_glob ~ sqrt(sum_i (c_i*E_i)^2)

Real deep networks sit between: successive layers neither perfectly align
nor perfectly cancel. `fit_aggregation` picks the exponent p in

    E_glob ~ (sum_i (c_i * E_i)^p)^(1/p)

from measured whole-model configurations, so the rule is calibrated rather
than assumed.

Once c_i is known the allocator's objective becomes sum_i c_i * E_i(r_i)
instead of sum_i E_i(r_i) -- same solver, correct weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InfluenceModel:
    """Per-operator influence coefficients plus the fitted aggregation rule."""

    coefficients: dict[str, float]
    p: float = 1.0                     # aggregation exponent
    scale: float = 1.0                 # global fitted scale factor

    def predict(self, local_errors: dict[str, float]) -> float:
        """Predicted end-to-end error for a configuration."""
        terms = [
            self.coefficients.get(name, 0.0) * err
            for name, err in local_errors.items()
        ]
        terms = [t for t in terms if t > 0]
        if not terms:
            return 0.0
        agg = float(np.sum(np.power(terms, self.p)) ** (1.0 / self.p))
        return self.scale * agg

    def weighted_errors(self, local_errors: dict[str, float]) -> dict[str, float]:
        """c_i * E_i -- what the allocator should actually minimize."""
        return {n: self.coefficients.get(n, 0.0) * e for n, e in local_errors.items()}


def influence_coefficients(
    single_op_global: dict[str, float],
    single_op_local: dict[str, float],
    eps: float = 1e-12,
) -> dict[str, float]:
    """c_i = E_glob(i alone) / E_loc(i), measured at the same perturbation."""
    return {
        name: float(single_op_global[name] / max(single_op_local.get(name, 0.0), eps))
        for name in single_op_global
    }


def fit_aggregation(
    coefficients: dict[str, float],
    configs: list[tuple[dict[str, float], float]],
    p_grid: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
) -> InfluenceModel:
    """Choose the aggregation exponent p (and a scale) that best explains the
    measured whole-model configurations.

    `configs`: list of (local_errors_by_operator, measured_global_error).
    Needs at least one config; with one it fits `scale` only, which still
    beats assuming a rule outright.
    """
    if not configs:
        return InfluenceModel(coefficients, p=1.0, scale=1.0)

    best = None
    for p in p_grid:
        preds, targets = [], []
        for local, measured in configs:
            m = InfluenceModel(coefficients, p=p, scale=1.0)
            preds.append(m.predict(local))
            targets.append(measured)
        preds_a, targets_a = np.array(preds), np.array(targets)
        denom = float(np.sum(preds_a * preds_a))
        scale = float(np.sum(preds_a * targets_a) / denom) if denom > 0 else 1.0
        resid = float(np.sum((scale * preds_a - targets_a) ** 2))
        if best is None or resid < best[0]:
            best = (resid, p, scale)
    _, p, scale = best
    return InfluenceModel(coefficients, p=p, scale=scale)


def relative_prediction_error(model: InfluenceModel, configs) -> list[float]:
    """|predicted - measured| / measured, per config -- the honesty check."""
    out = []
    for local, measured in configs:
        pred = model.predict(local)
        out.append(abs(pred - measured) / (abs(measured) + 1e-12))
    return out
