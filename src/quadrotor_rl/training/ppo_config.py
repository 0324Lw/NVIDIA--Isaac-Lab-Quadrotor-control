from __future__ import annotations

from typing import Any, Dict


def build_ppo_hparams(args: Any) -> Dict[str, Any]:
    return {
        "rollouts": int(args.rollouts),
        "learning_epochs": int(args.learning_epochs),
        "mini_batches": int(args.mini_batches),
        "learning_rate": float(args.lr),
        "discount_factor": float(args.discount_factor),
        "gae_lambda": float(args.gae_lambda),
        "kl_threshold": float(args.kl_threshold),
        "ratio_clip": float(args.ratio_clip),
        "value_clip": float(args.value_clip),
        "entropy_loss_scale": float(args.entropy_loss_scale),
        "value_loss_scale": float(args.value_loss_scale),
        "grad_norm_clip": float(args.grad_norm_clip),
    }
