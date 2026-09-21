"""The multi-vector pieces that the released ColNanoVDR towers depend on.

Each of these encodes a decision that is invisible in the numbers until it is
wrong: a projection bias, an unmasked special token, or weights that never get
folded in all produce a tower that trains fine and then retrieves differently
from the checkpoint it was measured as.
"""

import torch

from nanovdr import MultiVectorHead, Repr, mean_maxsim, score


def _unit(*shape):
    return torch.nn.functional.normalize(torch.randn(*shape), dim=-1)


def test_multivector_projection_is_bias_free():
    """A bias survives L2 normalisation as a direction every token is pulled
    towards, which is the collapse balanced transport exists to prevent."""
    head = MultiVectorHead(hidden=12, embed_dim=6)
    assert head.proj.bias is None


def test_multivector_head_emits_unit_tokens():
    head = MultiVectorHead(hidden=12, embed_dim=6)
    rep = head(torch.randn(2, 4, 12), torch.ones(2, 4, dtype=torch.bool))
    assert torch.allclose(rep.tokens.norm(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert rep.weights is None


def test_weight_head_reads_pre_projection_states():
    """The logits must come from the hidden states, not the projected tokens:
    the packaged module reproduces exactly that wiring."""
    head = MultiVectorHead(hidden=12, embed_dim=6, learn_weights=True)
    assert head.weight_head.in_features == 12
    rep = head(torch.randn(2, 4, 12), torch.ones(2, 4, dtype=torch.bool))
    assert rep.weights.shape == (2, 4)


def test_mean_maxsim_matches_the_explicit_loop():
    q, d = _unit(3, 5, 8), _unit(4, 7, 8)
    qm = torch.ones(3, 5, dtype=torch.bool)
    dm = torch.ones(4, 7, dtype=torch.bool)
    got = mean_maxsim(q, d, qm, dm)
    want = torch.zeros(3, 4)
    for i in range(3):
        for j in range(4):
            want[i, j] = (q[i] @ d[j].T).max(dim=-1).values.mean()
    assert torch.allclose(got, want, atol=1e-5)


def test_mean_maxsim_ignores_document_padding():
    q, d = _unit(2, 3, 8), _unit(2, 4, 8)
    qm = torch.ones(2, 3, dtype=torch.bool)
    full = mean_maxsim(q, d, qm, torch.ones(2, 4, dtype=torch.bool))
    padded = torch.cat([d, _unit(2, 2, 8)], dim=1)
    mask = torch.cat([torch.ones(2, 4, dtype=torch.bool), torch.zeros(2, 2, dtype=torch.bool)], dim=1)
    assert torch.allclose(full, mean_maxsim(q, padded, qm, mask), atol=1e-5)


def test_score_dispatches_on_both_geometries():
    qs = Repr(vec=_unit(2, 8))
    qm = Repr(tokens=_unit(2, 3, 8), mask=torch.ones(2, 3, dtype=torch.bool))
    ds = Repr(vec=_unit(5, 8))
    dm = Repr(tokens=_unit(5, 6, 8), mask=torch.ones(5, 6, dtype=torch.bool))
    assert score(qs, ds).shape == (2, 5)      # dot product
    assert score(qm, ds).shape == (2, 5)      # MaxSim over single-vector pages
    assert score(qm, dm).shape == (2, 5)      # full late interaction
