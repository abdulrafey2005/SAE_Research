
"""
Stage 5: SAE causal intervention engine.

Uses the same SAE configuration as Stage 3.
Intervenes on the final token position of Layer 12.
"""

from __future__ import annotations

import torch
from sae_lens import SAE


class SAEIntervention:

    def __init__(
        self,
        model,
        layer_index=12,
        sae_release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82"
    ):

        self.model = model

        self.sae, _, _ = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device="cuda"
        )

        self.sae.eval()

        self.mode = "noop"
        self.feature_id = None
        self.alpha = 0.0

        self.hook_calls = 0
        self.last_max_delta = 0.0

        self.handle = self.model.model.layers[
            layer_index
        ].register_forward_hook(self._hook)

    def configure(self, mode="noop", feature_id=None, alpha=0.0):

        allowed = {
            "disabled",
            "noop",
            "ablation",
            "steer_positive",
            "steer_negative"
            }

        if mode not in allowed:
            raise ValueError(f"Invalid intervention mode: {mode}")

        if feature_id is not None:
            if not 0 <= int(feature_id) < 16384:
                raise ValueError("Feature ID must be between 0 and 16383.")

        self.mode = mode
        self.feature_id = feature_id
        self.alpha = float(alpha)

        self.hook_calls = 0
        self.last_max_delta = 0.0

    def _hook(self, module, inputs, output):
        if self.mode == "disabled":
            return output
        hidden = output[0] if isinstance(output, tuple) else output

        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise RuntimeError(
                "Expected Layer-12 output with shape [batch, seq, hidden]."
            )

        # Final sequence position only.
        original = hidden[:, -1, :].float()

        # Encode residual activation into SAE latent space.
        latent = self.sae.encode(original)

        modified = latent

        if self.mode != "noop" and self.feature_id is not None:

            modified = latent.clone()

            fid = int(self.feature_id)

            if self.mode == "ablation":

                modified[:, fid] = 0.0

            elif self.mode == "steer_positive":
                modified[:, fid] += self.alpha

            elif self.mode == "steer_negative":
                modified[:, fid] = torch.clamp(
                    modified[:, fid] - self.alpha,
                    min=0.0
                )

        # No-op must produce exactly zero residual change.
        if self.mode == "noop":

            delta = torch.zeros_like(original)

        else:

            decoded_original = self.sae.decode(latent)
            decoded_modified = self.sae.decode(modified)

            delta = decoded_modified - decoded_original

        changed = hidden.clone()

        changed[:, -1, :] = (
            hidden[:, -1, :] + delta.to(hidden.dtype)
        )

        self.hook_calls += 1

        self.last_max_delta = float(
            delta.detach().abs().max().item()
        )

        if isinstance(output, tuple):
            return (changed,) + output[1:]

        return changed

    def close(self):

        if self.handle is not None:
            self.handle.remove()
            self.handle = None
