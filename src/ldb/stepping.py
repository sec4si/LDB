"""Synchronous GPT-2 layer debugger with in-memory replay checkpoints."""

from __future__ import annotations

import math
import shlex
from pathlib import Path
from typing import Any

from ldb.checkpoints import LayerCheckpoint, fingerprint, load_checkpoint, save_checkpoint
from ldb.debug_edits import ActivationEdit
from ldb.sae import SAEDictionary, SAEFeature

PatchState = tuple[dict[int, float], dict[str, ActivationEdit]]


class _Captured(Exception):
    pass


class LayerDebugger:
    """Own one exclusive, cache-free GPT-2 forward execution in eval mode."""

    def __init__(self, model: Any, inputs: Any, *, tokenizer: Any = None) -> None:
        import torch
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        if not isinstance(model, GPT2LMHeadModel) or model.training:
            raise ValueError("Layer debugging requires an eval-mode GPT2LMHeadModel")
        if model.config.add_cross_attention:
            raise ValueError("Cross-attention is not supported by layer replay")
        self.model: Any = model
        self.tokenizer = tokenizer
        self.input_ids = inputs["input_ids"].detach().cpu().clone()
        self.targets: tuple[int, int] | None = None
        self.index = 0
        self.hidden: Any = None
        self.args: Any = ()
        self.kwargs: Any = {}
        self.patches: dict[int, float] = {}
        self.edits: dict[str, ActivationEdit] = {}
        self._features: dict[ActivationEdit, SAEFeature] = {}
        self.undo_stack: list[PatchState] = []
        self.redo_stack: list[PatchState] = []
        self.snapshots: dict[str, Any] = {}
        self.breaks: dict[int, tuple[str, float]] = {}
        self.watches: dict[str, float] = {}
        self.feature_watches: dict[str, tuple[int, SAEDictionary, int, float]] = {}

        def capture(module: Any, args: Any, kwargs: Any) -> None:
            self.hidden = args[0].detach().clone()
            self.args = args[1:]
            self.kwargs = dict(kwargs)
            raise _Captured

        handle = model.transformer.h[0].register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.inference_mode():
                model(**inputs, use_cache=False)
        except _Captured:
            pass
        finally:
            handle.remove()
        if self.hidden is None:
            raise RuntimeError("No GPT-2 block executed")
        self.initial_shape: Any = self.hidden.shape
        self.initial_dtype: Any = self.hidden.dtype
        self.identity: str = self._identity()
        self.snapshot("start")

    def _identity(self) -> str:
        return fingerprint(
            {
                "config": self.model.config.to_dict(),
                "weights": self.model.state_dict(),
                "initial": self.snapshots["start"][1] if "start" in self.snapshots else self.hidden,
                "args": self.args,
                "kwargs": self.kwargs,
            }
        )

    def save(self, path: Path) -> None:
        if self._identity() != self.identity:
            raise ValueError("Model or forward context changed since session creation")
        self._validate_edits(self.edits)
        save_checkpoint(
            path,
            LayerCheckpoint(
                identity=self.identity,
                next_layer=self.index,
                patches=self.patches,
                edits=self.edits,
                activation_hash=fingerprint(self.hidden),
            ),
            self.hidden,
        )

    def load(self, path: Path) -> None:
        checkpoint, hidden = load_checkpoint(path)
        if checkpoint.identity != self.identity or self._identity() != self.identity:
            raise ValueError("Checkpoint model weights or input context do not match")
        if hidden.shape != self.initial_shape or hidden.dtype != self.initial_dtype:
            raise ValueError("Checkpoint activation shape or dtype does not match")
        count = len(self.model.transformer.h)
        if checkpoint.next_layer > count or any(
            not 0 <= layer < count or not math.isfinite(scale)
            for layer, scale in checkpoint.patches.items()
        ):
            raise ValueError("Invalid checkpoint layer or patch")
        features = self._validate_edits(checkpoint.edits)
        restored = hidden.to(self.hidden.device).clone()
        self.index, self.hidden = checkpoint.next_layer, restored
        self.patches = dict(checkpoint.patches)
        self.edits = dict(checkpoint.edits)
        self._features.update(features)
        self.undo_stack.clear()
        self.redo_stack.clear()

    def watch_feature(
        self, name: str, layer: int, path: Path, feature: int, threshold: float
    ) -> None:
        self._condition("norm", threshold)
        if not 0 <= layer < len(self.model.transformer.h):
            raise ValueError("Invalid feature watch layer")
        dictionary = SAEDictionary(path)
        dictionary.feature(feature)
        if dictionary.model_width != self.hidden.shape[-1]:
            raise ValueError("SAE dictionary width does not match residual stream")
        self.feature_watches[name] = layer, dictionary, feature, threshold

    def inspect(self) -> dict[str, Any]:
        import torch

        values = self.hidden.detach().float()
        return {
            "next_layer": self.index,
            "finished": self.index == len(self.model.transformer.h),
            "shape": list(values.shape),
            "norm": float(values.norm()),
            "max_abs": float(values.abs().max()),
            "nonfinite": int((~torch.isfinite(values)).sum()),
        }

    def step(self) -> dict[str, Any]:
        import torch

        if self.index >= len(self.model.transformer.h):
            raise ValueError("Execution is finished; restore a snapshot to replay")
        before = self.inspect()
        block = self.model.transformer.h[self.index]
        active = [edit for edit in self.edits.values() if edit.layer == self.index]

        def patch_heads(_: Any, inputs: Any) -> Any:
            value = inputs[0]
            for edit in active:
                if edit.kind == "head":
                    value = edit.apply(value, heads=self.model.config.n_head)
            return (value, *inputs[1:])

        handle = (
            block.attn.c_proj.register_forward_pre_hook(patch_heads)
            if any(edit.kind == "head" for edit in active)
            else None
        )
        try:
            with torch.inference_mode():
                value = block(self.hidden, *self.args, **self.kwargs)
                value = value * self.patches.get(self.index, 1.0)
                for edit in active:
                    if edit.kind != "head":
                        value = edit.apply(
                            value, heads=self.model.config.n_head, feature=self._features.get(edit)
                        )
                self.hidden = value
        finally:
            if handle is not None:
                handle.remove()
        self.index += 1
        after = self.inspect()
        after["watch_hits"] = [
            metric
            for metric, threshold in self.watches.items()
            if abs(after[metric] - before[metric]) > threshold
        ]
        features = {}
        for name, (layer, dictionary, feature, threshold) in self.feature_watches.items():
            if layer != self.index - 1:
                continue
            peak = float(dictionary.encode(self.hidden)[:, feature].max())
            features[name] = {"feature": feature, "peak": peak, "threshold": threshold}
            if peak > threshold or not math.isfinite(peak):
                after["watch_hits"].append(name)
        after["features"] = features
        return after

    def run(self) -> dict[str, Any]:
        while self.index < len(self.model.transformer.h):
            state = self.step()
            condition = self.breaks.get(self.index - 1)
            if condition is not None and state[condition[0]] > condition[1]:
                return {**state, "stop_reason": "breakpoint"}
            if state["watch_hits"]:
                return {**state, "stop_reason": "watch"}
        return self.inspect()

    def logits(self) -> Any:
        import torch

        if self.index != len(self.model.transformer.h):
            raise ValueError("Continue to the final layer before reading logits")
        with torch.inference_mode():
            return self.model.lm_head(self.model.transformer.ln_f(self.hidden))

    def snapshot(self, name: str) -> None:
        if name in self.snapshots:
            raise ValueError("Snapshot already exists")
        self.snapshots[name] = (self.index, self.hidden.detach().cpu().clone(), self._patch_state())

    def restore(self, name: str) -> None:
        index, hidden, state = self.snapshots[name]
        self.index = index
        self.hidden = hidden.to(self.hidden.device).clone()
        self._restore_patches(state)
        self.undo_stack.clear()
        self.redo_stack.clear()

    def patch(self, layer: int, scale: float | None) -> None:
        if not self.index <= layer < len(self.model.transformer.h):
            raise ValueError("Patch must target an unexecuted layer; restore first")
        if scale is not None and not math.isfinite(scale):
            raise ValueError("Scale must be finite")
        self.undo_stack.append(self._patch_state())
        self.redo_stack.clear()
        if scale is None:
            self.patches.pop(layer, None)
        else:
            self.patches[layer] = scale

    def history(self, redo: bool = False) -> None:
        source, target = (
            (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        )
        if not source:
            raise ValueError("No patch history available")
        proposed = source[-1]
        patches, edits = proposed
        changed = set(self.patches) | set(patches)
        if any(
            layer < self.index and self.patches.get(layer) != patches.get(layer)
            for layer in changed
        ) or any(
            edit.layer < self.index
            for name in set(self.edits) | set(edits)
            if self.edits.get(name) != edits.get(name)
            for edit in (self.edits.get(name), edits.get(name))
            if edit is not None
        ):
            raise ValueError("Cannot undo executed effects; restore a snapshot first")
        target.append(self._patch_state())
        self._restore_patches(source.pop())

    def branch(self, layer: int, scale: float) -> dict[str, Any]:
        """Run baseline and edited suffixes, then restore the paused state."""
        return self.compare(f"patch {layer} {scale}")

    def compare(self, command: str) -> dict[str, Any]:
        allowed = {"patch", "unpatch", "patch-token", "patch-head", "patch-feature", "unpatch-edit"}
        words = shlex.split(command)
        if not words or words[0] not in allowed:
            raise ValueError("Compare requires one patch or unpatch command")
        saved = self.index, self.hidden, self._patch_state()
        undo, redo = list(self.undo_stack), list(self.redo_stack)
        outputs = []
        try:
            self.command(command)
            edited = self._patch_state()
            for edit in (False, True):
                self.index, self.hidden, state = saved
                self._restore_patches(edited if edit else state)
                while self.index < len(self.model.transformer.h):
                    self.step()
                outputs.append(self.logits().detach().float().cpu())
        finally:
            self.index, self.hidden, state = saved
            self._restore_patches(state)
            self.undo_stack, self.redo_stack = undo, redo
        baseline, patched = (self._scores(output) for output in outputs)
        result = {
            "max_logit_delta": float((outputs[1] - outputs[0]).abs().max()),
            "baseline": baseline,
            "patched": patched,
        }
        if self.targets is not None:
            result["mean_margin_delta"] = patched["mean_margin"] - baseline["mean_margin"]
        return result

    def _patch_state(self) -> PatchState:
        return dict(self.patches), dict(self.edits)

    def _restore_patches(self, state: PatchState) -> None:
        self.patches, self.edits = dict(state[0]), dict(state[1])

    def _validate_edits(self, edits: dict[str, ActivationEdit]) -> dict[ActivationEdit, SAEFeature]:
        features = {}
        for edit in edits.values():
            edit.validate_context(self.model, self.hidden)
            if edit.kind == "feature":
                features[edit] = edit.load_feature(self.hidden.shape[-1])
        return features

    def edit(self, name: str, edit: ActivationEdit | None) -> None:
        for item in (self.edits.get(name), edit):
            if item is not None and item.layer < self.index:
                raise ValueError("Edit must target an unexecuted layer; restore first")
        resources = self._validate_edits({name: edit}) if edit is not None else {}
        self.undo_stack.append(self._patch_state())
        self.redo_stack.clear()
        if edit is None:
            self.edits.pop(name, None)
        else:
            self.edits[name] = edit
        self._features.update(resources)

    def target(self, answer: int, alternative: int) -> None:
        if answer == alternative or not all(
            0 <= token < self.model.config.vocab_size for token in (answer, alternative)
        ):
            raise ValueError("Targets must be distinct token IDs inside the vocabulary")
        self.targets = answer, alternative

    def target_text(self, answer: str, alternative: str) -> None:
        if self.tokenizer is None:
            raise ValueError("Text targets require a tokenizer; use target-ids instead")
        ids = []
        for text in (answer, alternative):
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("Each target must be exactly one token; preserve leading spaces")
            for row in self.input_ids.tolist():
                prompt = self.tokenizer.decode(row)
                if self.tokenizer.encode(prompt + text, add_special_tokens=False) != row + encoded:
                    raise ValueError("Target text changes the prompt token boundary")
            ids.append(encoded[0])
        self.target(*ids)

    def _scores(self, logits: Any) -> dict[str, Any]:
        values = logits[:, -1].detach().float().cpu()
        probabilities = values.softmax(-1)
        log_probabilities = values.log_softmax(-1)
        top = values.argmax(-1).tolist()
        result: dict[str, Any] = {"next_token_ids": top}
        if self.tokenizer is not None:
            result["next_tokens"] = [self.tokenizer.decode([token]) for token in top]
        if self.targets is not None:
            answer, alternative = self.targets
            margins = values[:, answer] - values[:, alternative]
            result.update(
                answer_token_id=answer,
                alternative_token_id=alternative,
                answer_probability=probabilities[:, answer].tolist(),
                alternative_probability=probabilities[:, alternative].tolist(),
                answer_log_probability=log_probabilities[:, answer].tolist(),
                alternative_log_probability=log_probabilities[:, alternative].tolist(),
                margin=margins.tolist(),
                mean_margin=float(margins.mean()),
                preference_rate=float((margins > 0).float().mean()),
            )
            if self.tokenizer is not None:
                result["answer"] = self.tokenizer.decode([answer])
                result["alternative"] = self.tokenizer.decode([alternative])
        return result

    def command(self, text: str) -> Any:
        words = shlex.split(text)
        match words:
            case ["step" | "next"]:
                return self.step()
            case ["continue"]:
                return self.run()
            case ["inspect"]:
                return self.inspect()
            case ["tokens"]:
                return [
                    [
                        {
                            "position": position,
                            "token_id": token,
                            "text": self.tokenizer.decode([token]) if self.tokenizer else None,
                        }
                        for position, token in enumerate(row)
                    ]
                    for row in self.input_ids.tolist()
                ]
            case ["snapshot", name]:
                self.snapshot(name)
            case ["restore", name]:
                self.restore(name)
            case ["save", path]:
                self.save(Path(path))
            case ["load", path]:
                self.load(Path(path))
            case ["snapshots"]:
                return {name: state[0] for name, state in self.snapshots.items()}
            case ["watch-feature", name, layer, path, feature, threshold]:
                self.watch_feature(name, int(layer), Path(path), int(feature), float(threshold))
            case ["unwatch-feature", name]:
                self.feature_watches.pop(name, None)
            case ["watches"]:
                return {
                    "residual": self.watches,
                    "features": {
                        name: {
                            "layer": item[0],
                            "dictionary_hash": item[1].digest,
                            "feature": item[2],
                            "threshold": item[3],
                        }
                        for name, item in self.feature_watches.items()
                    },
                }
            case ["breaks"]:
                return dict(self.breaks)
            case ["patch", layer, scale]:
                self.patch(int(layer), float(scale))
            case ["patch-token", name, layer, token, scale]:
                self.edit(
                    name,
                    ActivationEdit(
                        kind="token", layer=int(layer), token=int(token), scale=float(scale)
                    ),
                )
            case ["patch-head", name, layer, head, scale, *tokens] if len(tokens) <= 1:
                self.edit(
                    name,
                    ActivationEdit(
                        kind="head",
                        layer=int(layer),
                        head=int(head),
                        scale=float(scale),
                        token=int(tokens[0]) if tokens else None,
                    ),
                )
            case ["patch-feature", name, layer, path, feature, scale, *tokens] if len(tokens) <= 1:
                dictionary = SAEDictionary(Path(path).expanduser())
                self.edit(
                    name,
                    ActivationEdit(
                        kind="feature",
                        layer=int(layer),
                        dictionary=str(Path(path).expanduser().resolve()),
                        dictionary_hash=dictionary.digest,
                        feature=int(feature),
                        scale=float(scale),
                        token=int(tokens[0]) if tokens else None,
                    ),
                )
            case ["unpatch-edit", name]:
                self.edit(name, None)
            case ["edits"]:
                return {name: edit.model_dump(mode="json") for name, edit in self.edits.items()}
            case ["compare", command]:
                return self.compare(command)
            case ["regress", path, command]:
                from ldb.debug_evaluation import evaluate_debugger_suite

                return evaluate_debugger_suite(self, Path(path), command)
            case ["target", answer, alternative]:
                self.target_text(answer, alternative)
            case ["target-ids", answer, alternative]:
                self.target(int(answer), int(alternative))
            case ["scores"]:
                return self._scores(self.logits())
            case ["unpatch", layer]:
                self.patch(int(layer), None)
            case ["undo" | "redo" as operation]:
                self.history(redo=operation == "redo")
            case ["patches"]:
                return dict(self.patches)
            case ["branch", layer, scale]:
                return self.branch(int(layer), float(scale))
            case ["break", layer, metric, threshold]:
                self._condition(metric, float(threshold))
                if not 0 <= int(layer) < len(self.model.transformer.h):
                    raise ValueError("Invalid layer")
                self.breaks[int(layer)] = (metric, float(threshold))
            case ["watch", metric, threshold]:
                self._condition(metric, float(threshold))
                self.watches[metric] = float(threshold)
            case ["delete", layer]:
                self.breaks.pop(int(layer), None)
            case ["unwatch", metric]:
                self.watches.pop(metric, None)
            case ["logits"]:
                values = self.logits()[0, -1]
                return {"next_token_id": int(values.argmax()), "max_logit": float(values.max())}
            case _:
                raise ValueError("Unknown command; see docs/layer-debugger.md")
        return {"ok": True}

    @staticmethod
    def _condition(metric: str, threshold: float) -> None:
        if metric not in {"norm", "max_abs", "nonfinite"} or not math.isfinite(threshold):
            raise ValueError("Use norm, max_abs, or nonfinite with a finite threshold")
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
