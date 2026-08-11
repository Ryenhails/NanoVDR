"""The alignment registry is the one interface both towers depend on, so it is
the one thing worth testing without a GPU or a network."""

import pytest
import torch

from nanovdr import Repr, available_align_losses, get_align_loss


def _unit(*shape):
    return torch.nn.functional.normalize(torch.randn(*shape), dim=-1)


def test_registry_contents():
    assert set(available_align_losses()) == {"cosine", "ot", "chamfer", "coverage"}
    assert available_align_losses("single") == ["cosine"]
    assert set(available_align_losses("multi")) == {"ot", "chamfer", "coverage"}


def test_cosine_is_zero_at_the_optimum():
    v = _unit(4, 32)
    out = get_align_loss("cosine", geometry="single")(Repr(vec=v), Repr(vec=v))
    assert out["loss"].abs() < 1e-6


@pytest.mark.parametrize("name", ["ot", "chamfer", "coverage"])
def test_multi_objectives_are_lower_on_identical_sets(name):
    """Every set discrepancy must prefer a matching measure to a random one."""
    s, mask = _unit(2, 6, 32), torch.ones(2, 6, dtype=torch.bool)
    loss = get_align_loss(name, geometry="multi")
    same = loss(Repr(tokens=s, mask=mask), Repr(tokens=s, mask=mask))["loss"]
    other = loss(Repr(tokens=s, mask=mask), Repr(tokens=_unit(2, 6, 32), mask=mask))["loss"]
    assert same < other


@pytest.mark.parametrize("name", ["cosine", "ot", "chamfer", "coverage"])
def test_gradients_reach_the_student(name):
    geometry = "single" if name == "cosine" else "multi"
    if geometry == "single":
        s = _unit(3, 32).requires_grad_(True)
        student, teacher = Repr(vec=s), Repr(vec=_unit(3, 32))
    else:
        s = _unit(3, 5, 32).requires_grad_(True)
        mask = torch.ones(3, 5, dtype=torch.bool)
        student = Repr(tokens=s, mask=mask)
        teacher = Repr(tokens=_unit(3, 7, 32), mask=torch.ones(3, 7, dtype=torch.bool))
    get_align_loss(name, geometry=geometry)(student, teacher)["loss"].backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_masked_teacher_tokens_are_ignored():
    """Padding must not change the objective."""
    s, sm = _unit(1, 4, 16), torch.ones(1, 4, dtype=torch.bool)
    t = _unit(1, 6, 16)
    full = torch.ones(1, 6, dtype=torch.bool)
    padded = torch.cat([t, _unit(1, 3, 16)], dim=1)
    part = torch.cat([full, torch.zeros(1, 3, dtype=torch.bool)], dim=1)
    loss = get_align_loss("chamfer", geometry="multi")
    a = loss(Repr(tokens=s, mask=sm), Repr(tokens=t, mask=full))["loss"]
    b = loss(Repr(tokens=s, mask=sm), Repr(tokens=padded, mask=part))["loss"]
    assert torch.allclose(a, b, atol=1e-5)


def test_geometry_mismatch_is_refused():
    with pytest.raises(ValueError):
        get_align_loss("ot", geometry="single")
    with pytest.raises(ValueError):
        get_align_loss("cosine", geometry="single")(
            Repr(tokens=_unit(1, 3, 8), mask=torch.ones(1, 3, dtype=torch.bool)),
            Repr(tokens=_unit(1, 3, 8), mask=torch.ones(1, 3, dtype=torch.bool)),
        )


def test_repr_rejects_ambiguous_construction():
    with pytest.raises(ValueError):
        Repr()
    with pytest.raises(ValueError):
        Repr(vec=_unit(1, 4), tokens=_unit(1, 2, 4), mask=torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        Repr(tokens=_unit(1, 2, 4))          # missing mask
