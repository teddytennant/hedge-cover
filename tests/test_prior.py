"""Algorithm 1 prior and the Lemma C.2 style bound."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from hedge_cover import enumerate_hypotheses, prior


def _support(hypothesis) -> int:
    return sum(entry is not None for entry in hypothesis)


def _unnormalized(hypothesis, T, m) -> float:
    k = _support(hypothesis)
    return 1.0 / ((k + 1) ** 2 * (m + 1) ** k * math.comb(T, k))


def test_prior_cardinality_and_mass():
    T, m = 2, 2
    hypotheses = enumerate_hypotheses(T, m)
    assert len(hypotheses) == (m + 1) ** T == 9
    assert hypotheses[0] == (None, None)
    assert hypotheses[1] == (None, 0)
    assert hypotheses[-1] == (1, 1)

    p0 = prior(T, m)
    assert len(p0) == 9
    assert abs(sum(p0.values()) - 1.0) <= 1e-12
    assert all(weight > 0.0 for weight in p0.values())


def test_empty_support_outranks_full_support():
    T, m = 2, 2
    p0 = prior(T, m)
    empty = [weight for h, weight in p0.items() if _support(h) == 0]
    full = [weight for h, weight in p0.items() if _support(h) == 2]
    assert empty and full
    assert min(empty) > max(full)


def test_lemma_c2_bound_and_normalizer():
    T, m = 2, 2
    hypotheses = enumerate_hypotheses(T, m)
    weights = [_unnormalized(h, T, m) for h in hypotheses]
    Z = math.fsum(weights)
    Z_closed = sum((m / (m + 1)) ** k / (k + 1) ** 2 for k in range(T + 1))
    assert abs(Z - Z_closed) <= 1e-9
    assert Z <= math.pi**2 / 6 + 1e-9

    p0 = prior(T, m)
    for h, weight in zip(hypotheses, weights):
        assert abs(p0[h] - weight / Z) <= 1e-12

    def paper_rhs(k: int) -> float:
        extra = k * math.log(math.e * T * (m + 1) / k) if k > 0 else 0.0
        return 1.0 + 2.0 * math.log(k + 1) + extra + 1e-8

    def loose_rhs(k: int) -> float:
        extra = k * math.log(math.e * T * (m + 1) / max(k, 1))
        return math.log(Z) + 2.0 * math.log(k + 1) + extra + 1e-8

    log_inv = {h: math.log(1.0 / p0[h]) for h in hypotheses}
    paper_holds = all(
        log_inv[h] <= paper_rhs(_support(h)) for h in hypotheses
    )
    # Z <= pi^2/6 < e, so log(Z) <= 1 and the paper form is the right check.
    assert Z < math.e
    if paper_holds:
        for h in hypotheses:
            assert log_inv[h] <= paper_rhs(_support(h))
    else:
        for h in hypotheses:
            assert log_inv[h] <= loose_rhs(_support(h))
        assert Z <= math.pi**2 / 6 + 1e-9
    assert paper_holds
