from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .loss import actor_loss, belief_loss


@dataclass(frozen=True)
class QLoRAConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def prepare_qlora(model: Any, config: QLoRAConfig) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    prepared = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(prepared, peft_config)


def qlora_training_step(
    *,
    belief_logits: Any,
    belief_targets: Any,
    actor_logprob_fn: Any,
    rewards: Any,
    belief_state: Any,
    optimizer: Any,
    belief_weight: float = 1.0,
    actor_weight: float = 1.0,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    predictive = belief_loss(belief_logits, belief_targets)
    policy = actor_loss(actor_logprob_fn, rewards, belief_state)
    total = belief_weight * predictive + actor_weight * policy
    total.backward()
    optimizer.step()
    return {
        "belief_loss": float(predictive.detach().cpu()),
        "actor_loss": float(policy.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
