"""Hugging Face eager-mode runtime adapter."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from typing import Any, cast

from ldb.identity import hash_tokenizer
from ldb.models import HookPhase, HookPoint
from ldb.runtime import InterventionCallback, ProbeCallback, ProbeHandle


def _first_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _replace_first_tensor(value: Any, replacement: Any) -> tuple[Any, bool]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.shape != replacement.shape:
            raise ValueError(
                f"Replacement shape {tuple(replacement.shape)} does not match {tuple(value.shape)}"
            )
        return replacement.to(device=value.device, dtype=value.dtype), True
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            items[index], replaced = _replace_first_tensor(item, replacement)
            if replaced:
                return tuple(items), True
    elif isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            items[index], replaced = _replace_first_tensor(item, replacement)
            if replaced:
                return items, True
    return value, False


_FINAL_NORMS = ("transformer.ln_f", "model.norm", "model.model.norm")


class TransformersAdapter:
    """Normalize common decoder-only Hugging Face model topologies."""

    adapter_version = "0.1.0.dev0"

    def __init__(self, model_id: str, *, revision: str | None = None, device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_id = model_id
        self.revision = revision
        self.device = device
        tokenizer_factory = cast(Any, AutoTokenizer)
        model_factory = cast(Any, AutoModelForCausalLM)
        self.tokenizer: Any = tokenizer_factory.from_pretrained(model_id, revision=revision)
        self.model: Any = model_factory.from_pretrained(model_id, revision=revision)
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.model.to(device)
        self.model.eval()
        self._module_by_path = dict(self.model.named_modules())
        self._hooks = self._discover_hooks()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend(self) -> str:
        return "transformers"

    @property
    def backend_version(self) -> str:
        return version("transformers")

    @property
    def tokenizer_hash(self) -> str:
        return hash_tokenizer(self.tokenizer)

    def _block_path(self) -> str:
        candidates = ("transformer.h", "model.layers", "model.model.layers")
        for path in candidates:
            module = self._module_by_path.get(path)
            if module is not None and hasattr(module, "__len__"):
                return path
        raise ValueError(
            "Unsupported model topology: expected transformer.h, model.layers, "
            "or model.model.layers"
        )

    def _discover_hooks(self) -> tuple[HookPoint, ...]:
        block_path = self._block_path()
        blocks = self._module_by_path[block_path]
        hooks: list[HookPoint] = []
        for index in range(len(blocks)):
            prefix = f"{block_path}.{index}"
            hooks.extend(
                (
                    HookPoint(
                        name=f"block.{index}.residual.pre",
                        module_path=prefix,
                        phase=HookPhase.PRE,
                    ),
                    HookPoint(
                        name=f"block.{index}.residual.post",
                        module_path=prefix,
                        phase=HookPhase.POST,
                    ),
                )
            )
            for normalized, attributes in (
                ("attention.output", ("self_attn", "attn")),
                ("mlp.output", ("mlp",)),
            ):
                module_path = next(
                    (
                        f"{prefix}.{attribute}"
                        for attribute in attributes
                        if f"{prefix}.{attribute}" in self._module_by_path
                    ),
                    None,
                )
                if module_path:
                    hooks.append(
                        HookPoint(
                            name=f"block.{index}.{normalized}",
                            module_path=module_path,
                            phase=HookPhase.POST,
                        )
                    )
            projection_path = next(
                (
                    path
                    for path in (f"{prefix}.self_attn.o_proj", f"{prefix}.attn.c_proj")
                    if path in self._module_by_path
                ),
                None,
            )
            if projection_path is not None:
                hooks.append(
                    HookPoint(
                        name=f"block.{index}.attention.heads",
                        module_path=projection_path,
                        phase=HookPhase.PRE,
                    )
                )
            experts_path = next(
                (
                    path
                    for path in (
                        f"{prefix}.mlp.experts",
                        f"{prefix}.block_sparse_moe.experts",
                    )
                    if path in self._module_by_path
                    and hasattr(self._module_by_path[path], "__len__")
                ),
                None,
            )
            if experts_path is not None:
                for expert in range(len(self._module_by_path[experts_path])):
                    hooks.append(
                        HookPoint(
                            name=f"block.{index}.moe.expert.{expert}.output",
                            module_path=f"{experts_path}.{expert}",
                            phase=HookPhase.POST,
                        )
                    )
        return tuple(hooks)

    def hook_points(self) -> Sequence[HookPoint]:
        return self._hooks

    def install_probe(self, hook: HookPoint, callback: ProbeCallback) -> ProbeHandle:
        module = self._module_by_path[hook.module_path]

        if hook.phase is HookPhase.PRE:

            def pre_hook(_: Any, inputs: tuple[Any, ...]) -> None:
                tensor = _first_tensor(inputs)
                if tensor is not None:
                    callback(hook, tensor)

            return cast(ProbeHandle, module.register_forward_pre_hook(pre_hook))

        def post_hook(_: Any, __: tuple[Any, ...], output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                callback(hook, tensor)

        return cast(ProbeHandle, module.register_forward_hook(post_hook))

    def install_intervention(self, hook: HookPoint, callback: InterventionCallback) -> ProbeHandle:
        module = self._module_by_path[hook.module_path]

        if hook.phase is HookPhase.PRE:

            def pre_hook(_: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                tensor = _first_tensor(inputs)
                if tensor is None:
                    return inputs
                replacement = callback(hook, tensor)
                updated, replaced = _replace_first_tensor(inputs, replacement)
                return cast(tuple[Any, ...], updated if replaced else inputs)

            return cast(ProbeHandle, module.register_forward_pre_hook(pre_hook))

        def post_hook(_: Any, __: tuple[Any, ...], output: Any) -> Any:
            tensor = _first_tensor(output)
            if tensor is None:
                return output
            replacement = callback(hook, tensor)
            updated, replaced = _replace_first_tensor(output, replacement)
            return updated if replaced else output

        return cast(ProbeHandle, module.register_forward_hook(post_hook))

    def tokenize(self, prompt: str) -> Sequence[int]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return tuple(int(token) for token in encoded["input_ids"][0])

    def unembed(self, residual: Any) -> Any:
        """Apply the final norm and output projection to a residual-stream tensor."""

        import torch

        norm_path = next((path for path in _FINAL_NORMS if path in self._module_by_path), None)
        if norm_path is None:
            raise ValueError(
                "Unsupported model topology: expected transformer.ln_f, model.norm, "
                "or model.model.norm"
            )
        with torch.inference_mode():
            hidden = self._module_by_path[norm_path](residual.to(self.device))
            return self.model.get_output_embeddings()(hidden)

    def decode(self, token_id: int) -> str:
        return str(self.tokenizer.decode([token_id]))

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return str(self.tokenizer.decode(output[0], skip_special_tokens=True))

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**encoded).logits[:, -1, :]
            if target_token_id < 0 or target_token_id >= logits.shape[-1]:
                raise ValueError(
                    f"Target token ID {target_token_id} is outside the model vocabulary"
                )
            return float(torch.log_softmax(logits.float(), dim=-1)[0, target_token_id].item())

    def continuation_log_probability(self, prompt: str, continuation: str) -> float:
        import torch

        prompt_ids = list(self.tokenizer(prompt)["input_ids"])
        combined_ids = list(self.tokenizer(prompt + continuation)["input_ids"])
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
            logits = self.model(input_ids=input_ids).logits[0]
            positions = torch.arange(
                len(prompt_ids) - 1,
                len(prompt_ids) + len(continuation_ids) - 1,
                device=logits.device,
            )
            targets = torch.tensor(continuation_ids, device=logits.device, dtype=torch.long)
            values = torch.log_softmax(logits[positions].float(), dim=-1)
            return float(values.gather(1, targets[:, None]).mean().item())
