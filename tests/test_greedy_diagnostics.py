"""The instrumented reranker used by the robustness analyses must reproduce MV-EAC exactly
under the paper's rule for unlabeled articles, and the alternative rule must only differ
through the exploration bonus of unlabeled candidates."""
from __future__ import annotations

import numpy as np

from mveac.analysis.greedy_diagnostics import diagnostic_multi_view_eac
from mveac.calibration.eac import multi_view_eac

VOCABS = {"cat": ["a", "b", "c"], "ent": ["x", "y", "z"]}
MAPS = {
    "cat": {1: ["a"], 2: ["b"], 3: ["a"], 4: ["c"], 5: ["b"], 6: ["a"]},
    "ent": {1: ["x", "y"], 2: [], 3: ["x"], 4: [], 5: ["z"], 6: ["y", "y"]},
}
CANDIDATES = [(1, 0.9), (2, 0.8), (3, 0.75), (4, 0.4), (5, 0.35), (6, 0.1)]
PROFILES = {"cat": np.array([0.6, 0.3, 0.1]), "ent": np.array([0.5, 0.4, 0.1])}
WEIGHTS = {"cat": 0.5, "ent": 0.5}


def test_zero_rule_reproduces_multi_view_eac():
    for lam, beta in [(0.1, 0.0), (0.5, 0.9), (0.9, 2.0)]:
        expected = multi_view_eac(CANDIDATES, PROFILES, MAPS, VOCABS, WEIGHTS, lam, beta, 4)
        got = diagnostic_multi_view_eac(CANDIDATES, PROFILES, MAPS, VOCABS, WEIGHTS, lam, beta, 4)
        assert [a for a, _ in got] == [a for a, _ in expected]
        np.testing.assert_allclose([s for _, s in got], [s for _, s in expected])


def test_rules_coincide_without_exploration():
    zero = diagnostic_multi_view_eac(CANDIDATES, PROFILES, MAPS, VOCABS, WEIGHTS, 0.5, 0.0, 4)
    maxr = diagnostic_multi_view_eac(CANDIDATES, PROFILES, MAPS, VOCABS, WEIGHTS, 0.5, 0.0, 4, unlabeled_rule="max")
    assert zero == maxr


def test_step_records_have_one_flip_indicator_per_view():
    _, steps = diagnostic_multi_view_eac(CANDIDATES, PROFILES, MAPS, VOCABS, WEIGHTS, 0.5, 0.9, 4, record_steps=True)
    assert steps and all({"flip_cat", "flip_ent", "sd_rel"} <= set(s) for s in steps)
