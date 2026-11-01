"""TransformerLens 3.x bridge adapter for LDB offline analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib.metadata import version
from typing import Any

from ldb.identity import hash_tokenizer
from ldb.models import HookPhase, HookPoint, TensorAxis
from ldb.runtime import InterventionCallback, ProbeCallback, ProbeHandle


class _LensProbeHandle:
    def __init__(
        self, hook_point: Any, lens_handle: Any, on_remove: Callable[[], None] | None = None
    ) -> None:
        self._hook_point = hook_point
        self._lens_handle = lens_handle
        self._on_remove = on_remove

    def remove(self) -> None:
        if self._lens_handle is None:
            return
        self._lens_handle.hook.remove()
        if self._lens_handle in self._hook_point.fwd_hooks:
            self._hook_point.fwd_hooks.remove(self._lens_handle)
        self._lens_handle = None
        if self._on_remove is not None:
            self._on_remove()
            self._on_remove = None


class TransformerLensAdapter:
    """Expose TransformerBridge through LDB's causal runtime contract."""

    adapter_version = "0.1.0.dev0"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        _bridge: Any = None,
        _backend_version: str | None = None,
    ) -> None:
        if _bridge is None:
            try:
                from transformer_lens.model_bridge import TransformerBridge
            except ImportError as exc:
                raise RuntimeError(
                    "The TransformerLens engine requires "
                    "`python -m pip install 'latent-debugger[interpretability]'`"
                ) from exc

            _bridge = TransformerBridge.boot_transformers(
                model_id,
                device=device,
                revision=revision,
            )
        self._model_id = model_id
        self.model = _bridge
        self.tokenizer = _bridge.tokenizer
        self.device = device
        self._backend_version = _backend_version or version("transformer-lens")
        config = getattr(getattr(_bridge, "original_model", None), "config", None)
        if bool(getattr(config, "is_encoder_decoder", False)):
            raise ValueError("LDB offline causal workflows require a decoder-only model")
        if not bool(getattr(getattr(_bridge, "adapter", None), "supports_generation", True)):
            raise ValueError("LDB offline causal workflows require an autoregressive model")
        self.revision = getattr(config, "_commit_hash", None) or revision
        self._result_hook_count = 0
        self._result_initially_enabled = bool(
            getattr(getattr(self.model, "cfg", None), "use_attn_result", False)
        )
        self._hooks = self._discover_hooks()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend(self) -> str:
        return "transformerlens"

    @property
    def backend_version(self) -> str:
        return self._backend_version

    @property
    def tokenizer_hash(self) -> str:
        return hash_tokenizer(self.tokenizer)

    def _discover_hooks(self) -> tuple[HookPoint, ...]:
        available = self.model.hook_dict
        layers = sorted(
            {
                int(parts[1])
                for name in available
                if len(parts := name.split(".")) >= 3
                and parts[0] == "blocks"
                and parts[1].isdigit()
            }
        )
        hooks: list[HookPoint] = []
        stream = (TensorAxis.BATCH, TensorAxis.TOKEN, TensorAxis.CHANNEL)
        per_head = (
            TensorAxis.BATCH,
            TensorAxis.TOKEN,
            TensorAxis.HEAD,
            TensorAxis.CHANNEL,
        )
        attention_map = (
            TensorAxis.BATCH,
            TensorAxis.HEAD,
            TensorAxis.QUERY_TOKEN,
            TensorAxis.KEY_TOKEN,
        )
        for layer in layers:
            prefix = f"blocks.{layer}"
            specifications = (
                (
                    "residual.pre",
                    (f"{prefix}.hook_in", f"{prefix}.hook_resid_pre"),
                    stream,
                ),
                (
                    "residual.mid",
                    (f"{prefix}.hook_resid_mid", f"{prefix}.ln2.hook_in"),
                    stream,
                ),
                ("attention.q", (f"{prefix}.attn.hook_q",), per_head),
                ("attention.k", (f"{prefix}.attn.hook_k",), per_head),
                ("attention.v", (f"{prefix}.attn.hook_v",), per_head),
                ("attention.scores", (f"{prefix}.attn.hook_attn_scores",), attention_map),
                ("attention.pattern", (f"{prefix}.attn.hook_pattern",), attention_map),
                ("attention.z", (f"{prefix}.attn.hook_z",), per_head),
                ("attention.result", (f"{prefix}.attn.hook_result",), per_head),
                ("attention.output", (f"{prefix}.attn.hook_out",), stream),
                ("mlp.input", (f"{prefix}.mlp.hook_in",), stream),
                (
                    "mlp.pre",
                    (f"{prefix}.mlp.hook_pre", f"{prefix}.mlp.in.hook_out"),
                    stream,
                ),
                (
                    "mlp.activation",
                    (f"{prefix}.mlp.hook_post", f"{prefix}.mlp.out.hook_in"),
                    stream,
                ),
                ("mlp.output", (f"{prefix}.mlp.hook_out",), stream),
                (
                    "residual.post",
                    (f"{prefix}.hook_out", f"{prefix}.hook_resid_post"),
                    stream,
                ),
            )
            for normalized, candidates, axes in specifications:
                source = next((name for name in candidates if name in available), None)
                if source is not None:
                    hooks.append(
                        HookPoint(
                            name=f"block.{layer}.{normalized}",
                            module_path=source,
                            phase=HookPhase.POST,
                            axes=axes,
                        )
                    )
        if not hooks:
            raise ValueError("TransformerBridge exposes no supported decoder block hooks")
        return tuple(hooks)

    def hook_points(self) -> Sequence[HookPoint]:
        return self._hooks

    def install_probe(self, hook: HookPoint, callback: ProbeCallback) -> ProbeHandle:
        hook_point = self.model.hook_dict[hook.module_path]
        hook_point_spec = hook
        release = self._prepare_optional_hook(hook)

        def probe(tensor: Any, *, hook: Any) -> None:
            del hook
            callback(hook_point_spec, tensor)

        try:
            hook_point.add_hook(probe)
        except Exception:
            release()
            raise
        return _LensProbeHandle(hook_point, hook_point.fwd_hooks[-1], release)

    def install_intervention(self, hook: HookPoint, callback: InterventionCallback) -> ProbeHandle:
        hook_point = self.model.hook_dict[hook.module_path]
        hook_point_spec = hook
        release = self._prepare_optional_hook(hook)

        def intervene(tensor: Any, *, hook: Any) -> Any:
            del hook
            return callback(hook_point_spec, tensor)

        try:
            hook_point.add_hook(intervene)
        except Exception:
            release()
            raise
        return _LensProbeHandle(hook_point, hook_point.fwd_hooks[-1], release)

    def _prepare_optional_hook(self, hook: HookPoint) -> Callable[[], None]:
        if not hook.name.endswith(".attention.result"):
            return lambda: None
        if self._result_hook_count == 0 and not self._result_initially_enabled:
            self.model.set_use_attn_result(True)
        self._result_hook_count += 1

        def release() -> None:
            self._result_hook_count -= 1
            if self._result_hook_count == 0 and not self._result_initially_enabled:
                self.model.set_use_attn_result(False)

        return release

    def tokenize(self, prompt: str) -> Sequence[int]:
        tokens = self.model.to_tokens(
            prompt,
            prepend_bos=False,
            move_to_device=False,
        )
        return tuple(int(token) for token in tokens[0])

    def unembed(self, residual: Any) -> Any:
        """Apply the final norm and unembedding to a residual-stream tensor."""

        import torch

        with torch.inference_mode():
            return self.model.unembed(self.model.ln_final(residual))

    def decode(self, token_id: int) -> str:
        return str(self.tokenizer.decode([token_id]))

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        output = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            prepend_bos=False,
            return_type="str",
            verbose=False,
        )
        if not isinstance(output, str):
            raise TypeError("TransformerBridge returned a non-string generation")
        return output

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        import torch

        with torch.inference_mode():
            logits = self.model(prompt, return_type="logits", prepend_bos=False)[:, -1, :]
            if target_token_id < 0 or target_token_id >= logits.shape[-1]:
                raise ValueError(
                    f"Target token ID {target_token_id} is outside the model vocabulary"
                )
            return float(torch.log_softmax(logits.float(), dim=-1)[0, target_token_id].item())

    def continuation_log_probability(self, prompt: str, continuation: str) -> float:
        import torch

        prompt_ids = list(self.tokenize(prompt))
        combined_ids = list(self.tokenize(prompt + continuation))
        if not prompt_ids:
            raise ValueError("Prompt must contain at least one token")
        if combined_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Continuation changes the prompt token boundary; include an explicit separator"
            )
        continuation_ids = combined_ids[len(prompt_ids) :]
        if not continuation_ids:
            raise ValueError("Continuation must contain at least one token")
        input_ids = torch.tensor([combined_ids], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(input_ids, return_type="logits")[0]
            positions = torch.arange(
                len(prompt_ids) - 1,
                len(prompt_ids) + len(continuation_ids) - 1,
                device=logits.device,
            )
            targets = torch.tensor(continuation_ids, device=logits.device, dtype=torch.long)
            values = torch.log_softmax(logits[positions].float(), dim=-1)
            return float(values.gather(1, targets[:, None]).mean().item())
