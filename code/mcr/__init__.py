"""mcr — Markov-Chain Reliability toolkit for absorbing agent chains.

Public API:
    reliability(Q, R_succ, s0=0, d=None)
    asymptotic_reliability(Q, R_succ, s0=0)
    fundamental_matrix(Q)
    monte_carlo_reliability(Q, R_succ, R_fail, s0=0, d=None, n=10_000, rng=None)
    perturb(Q0, R0, eps, delta, reroute_to='fail')
    random_substochastic(m, rho_target=0.7, density=1.0, rng=None)
    nhpp_scaling_family(m, eps_seq, rng=None)
"""
from .reliability import (
    reliability,
    asymptotic_reliability,
    fundamental_matrix,
)
from .simulate import monte_carlo_reliability
from .perturb import perturb
from .chains import random_substochastic, nhpp_scaling_family
from .trace_to_chain import (
    TraceStep,
    ChainFit,
    fit as fit_chain,
    first_passage_times,
    empirical_first_passage_from_traces,
    goodness_of_fit,
)

__all__ = [
    "reliability",
    "asymptotic_reliability",
    "fundamental_matrix",
    "monte_carlo_reliability",
    "perturb",
    "random_substochastic",
    "nhpp_scaling_family",
    "TraceStep",
    "ChainFit",
    "fit_chain",
    "first_passage_times",
    "empirical_first_passage_from_traces",
    "goodness_of_fit",
]
__version__ = "0.2.0"
