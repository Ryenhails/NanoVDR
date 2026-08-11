"""Multi-vector alignment objectives.

The student emits a set of token vectors and the teacher emits another set, on
the unit sphere, with no correspondence between them and generally different
cardinalities. The loss is therefore a discrepancy between two discrete
measures rather than between two points.

    mu_S = sum_i a_i delta_{s_i},    mu_T = sum_j b_j delta_{t_j}
    cost  c(s, t) = 1 - <s, t>

The single-vector cosine objective is the one-atom case of exactly this.

Entropic optimal transport (``ot``) is the default. Chamfer and its one-sided
relaxation are kept because they are the ablations that motivate it: coverage
alone leaves dead student tokens, and adding precision collapses the student
onto a few teacher atoms. Balanced transport removes both failure modes by
enforcing the marginals on each side.

None of these objectives touches documents. Query-side token sets and their
cached teacher targets are all they consume.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import AlignLoss, Repr, register_align_loss

__all__ = ["SinkhornOTAlignLoss", "ChamferAlignLoss", "CoverageAlignLoss", "sinkhorn_log"]


def sinkhorn_log(
    C: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    n_iter: int,
) -> torch.Tensor:
    """Batched log-domain Sinkhorn, returning the entropic transport plan.

    Parameters
    ----------
    C : (B, Ks, Kt)
        Cost matrix, finite everywhere; masked entries should already be 0.
    log_a : (B, Ks)
        Log row marginals, ``-inf`` at invalid student tokens.
    log_b : (B, Kt)
        Log column marginals, ``-inf`` at invalid teacher tokens.

    The potentials are iterated under ``no_grad`` (envelope theorem) and one
    final differentiable step is taken, so gradients through the returned plan
    are correct at the fixed point without backpropagating through every
    iteration.
    """

    def _step(f, g):
        g_e = g.unsqueeze(1)                                     # (B, 1, Kt)
        f = eps * log_a - eps * torch.logsumexp((g_e - C) / eps, dim=2)
        f_e = f.unsqueeze(2)                                     # (B, Ks, 1)
        g = eps * log_b - eps * torch.logsumexp((f_e - C) / eps, dim=1)
        return f, g

    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    with torch.no_grad():
        for _ in range(n_iter):
            f, g = _step(f, g)
    f, g = _step(f, g)
    return torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - C) / eps)


def _uniform_log_marginal(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Log of a uniform distribution over the valid entries of ``mask``."""
    neg_inf = torch.finfo(torch.float32).min
    base = torch.where(mask, torch.zeros_like(ref[..., 0]), torch.full_like(ref[..., 0], neg_inf))
    return base - torch.logsumexp(base, dim=1, keepdim=True)


@register_align_loss(
    "ot",
    geometry="multi",
    summary="entropic Sinkhorn transport between the student and teacher token measures",
)
class SinkhornOTAlignLoss(AlignLoss):
    """Balanced entropic optimal transport on the unit sphere.

    ``L = <P*, C>`` where ``P*`` minimises ``<P, C> - eps H(P)`` over couplings
    with the prescribed marginals. Enforcing both marginals is what makes this
    behave: every student token must carry its mass, so none goes dead, and the
    student has to spread to match the teacher's measure rather than collapsing.

    Parameters
    ----------
    eps : float
        Entropic regularisation strength.
    n_iter : int
        Sinkhorn iterations.
    weighted : bool
        If True, the student marginal is ``softmax`` over per-token weight
        logits (``Repr.weights``) instead of uniform, letting a student token
        carry more mass when it has fewer atoms than the teacher. At inference
        the tokens must then be scaled by the same softmax weights before
        MaxSim.
    divergence : bool
        If True, return the debiased Sinkhorn divergence
        ``OT(mu_S, mu_T) - 0.5 OT(mu_S, mu_S)``. The teacher self-term is
        constant in the parameters and dropped. This removes the entropic
        shrinkage bias so the objective reaches 0 at ``mu_S = mu_T``.
    """

    def __init__(
        self,
        eps: float = 0.05,
        n_iter: int = 50,
        weighted: bool = False,
        divergence: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.n_iter = n_iter
        self.weighted = weighted
        self.divergence = divergence

    def _ot(self, s, sm, t, tm, log_a, log_b) -> torch.Tensor:
        sim = torch.einsum("bid,bjd->bij", s.float(), t.float())
        C = (1.0 - sim).masked_fill(~(sm.unsqueeze(2) & tm.unsqueeze(1)), 0.0)
        P = sinkhorn_log(C, log_a, log_b, self.eps, self.n_iter)
        return (P * C).sum(dim=(1, 2))

    def compute(self, student: Repr, teacher: Repr) -> dict[str, torch.Tensor]:
        s, sm, t, tm = student.tokens, student.mask, teacher.tokens, teacher.mask
        log_b = _uniform_log_marginal(tm, t)
        if self.weighted and student.weights is not None:
            neg_inf = torch.finfo(torch.float32).min
            sw = student.weights.float().masked_fill(~sm, neg_inf)
            log_a = sw - torch.logsumexp(sw, dim=1, keepdim=True)
        else:
            log_a = _uniform_log_marginal(sm, s)

        cost = self._ot(s, sm, t, tm, log_a, log_b)
        out = {"ot": cost.mean().detach()}
        if self.divergence:
            self_cost = self._ot(s, sm, s, sm, log_a, log_a)
            out["ot_self"] = self_cost.mean().detach()
            out["loss"] = (cost - 0.5 * self_cost).mean()
        else:
            out["loss"] = cost.mean()
        return out


@register_align_loss(
    "chamfer",
    geometry="multi",
    summary="bidirectional Chamfer distance between the two token sets (ablation)",
)
class ChamferAlignLoss(AlignLoss):
    """Bidirectional Chamfer distance between the token sets.

        coverage  = mean_j [ 1 - max_i <s_i, t_j> ]   every teacher token covered
        precision = mean_i [ 1 - max_j <s_i, t_j> ]   every student token grounded

    Unlike score-based objectives, whose gradient flows only through an
    aggregated argmax, this gives dense per-token supervision. It is kept as an
    ablation: adding the precision side collapses the student, which is the
    failure balanced transport avoids.

    ``direction`` is one of ``"both"``, ``"coverage"``, ``"precision"``.
    """

    def __init__(self, direction: str = "both"):
        super().__init__()
        if direction not in ("both", "coverage", "precision"):
            raise ValueError(f"direction must be both/coverage/precision, got {direction!r}")
        self.direction = direction

    def compute(self, student: Repr, teacher: Repr) -> dict[str, torch.Tensor]:
        s, sm, t, tm = student.tokens, student.mask, teacher.tokens, teacher.mask
        sim = torch.einsum("bid,bjd->bij", s.float(), t.float())
        pair = sm.unsqueeze(2) & tm.unsqueeze(1)
        neg_inf = torch.finfo(torch.float32).min
        sim_m = sim.masked_fill(~pair, neg_inf)

        out: dict[str, torch.Tensor] = {}
        terms = []
        if self.direction in ("both", "coverage"):
            best_per_teacher = sim_m.max(dim=1).values                      # (B, Kt)
            cov = (1.0 - best_per_teacher).masked_fill(~tm, 0.0).sum(1) / tm.sum(1).clamp(min=1)
            out["coverage"] = cov.mean().detach()
            terms.append(cov)
        if self.direction in ("both", "precision"):
            best_per_student = sim_m.max(dim=2).values                      # (B, Ks)
            prec = (1.0 - best_per_student).masked_fill(~sm, 0.0).sum(1) / sm.sum(1).clamp(min=1)
            out["precision"] = prec.mean().detach()
            terms.append(prec)

        out["loss"] = torch.stack(terms, dim=0).mean(0).mean()
        return out


@register_align_loss(
    "coverage",
    geometry="multi",
    summary="one-sided Chamfer: every teacher token must be covered (ablation)",
)
class CoverageAlignLoss(ChamferAlignLoss):
    """One-sided relaxation of transport: cover the teacher, ignore precision.

    Cheaper than OT and a reasonable first thing to try, but it leaves student
    tokens that no teacher token pulls on, so they drift. Kept as the ablation
    that motivates balanced transport.
    """

    def __init__(self):
        super().__init__(direction="coverage")
