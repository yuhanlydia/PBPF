"""Association-aware particle belief filtering primitives."""

from .acquisition import (
    AcquisitionTrace,
    ActiveTestPolicy,
    DiagnosticParticleState,
    NeuralDiagnosticAdapter,
    TestDecision,
    expected_information_gain,
)
from .association import association_gap, association_margin_loss
from .counterfactual import association_eligibility, joint_permutation, outcome_derangement
from .hard_bank import (
    HardBankAudit,
    HardBankPopulationLock,
    audit_hard_bank,
    lock_hard_bank_population,
    select_development_eligible_groups,
)

__all__ = [
    "AcquisitionTrace",
    "ActiveTestPolicy",
    "DiagnosticParticleState",
    "HardBankAudit",
    "HardBankPopulationLock",
    "NeuralDiagnosticAdapter",
    "TestDecision",
    "association_eligibility",
    "association_gap",
    "association_margin_loss",
    "audit_hard_bank",
    "expected_information_gain",
    "joint_permutation",
    "lock_hard_bank_population",
    "outcome_derangement",
    "select_development_eligible_groups",
]
