"""Exact upstream Hadamard reference matrix for correctness/calibration work."""

import ast
import math

import torch


def hadamard_matrix(source, size, device):
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    scope = dict(torch=torch, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope["matmul_hadU"](torch.eye(size, device=device, dtype=torch.float64))


@torch.inference_mode()
def rotate_residual(model, source, seed=20260911):
    """Fuse Mixtral RMSNorm scales and rotate the entire residual coordinate basis."""
    if model.config.model_type != "mixtral" or model.config.tie_word_embeddings:
        raise ValueError("Expected untied Mixtral model")
    device = model.model.embed_tokens.weight.device
    q = hadamard_matrix(source, model.config.hidden_size, device)
    rng = torch.Generator(device=device).manual_seed(seed)
    q = (torch.randint(0, 2, (q.shape[0],), device=device, generator=rng) * 2 - 1)[:, None] * q

    def right(weight, scale=None):
        value = weight.double()
        if scale is not None:
            value = value * scale.double()
        weight.copy_((value @ q).to(weight.dtype))

    def left(weight):
        weight.copy_((q.T @ weight.double()).to(weight.dtype))

    right(model.model.embed_tokens.weight)
    right(model.lm_head.weight, model.model.norm.weight)
    model.model.norm.weight.fill_(1)
    for layer in model.model.layers:
        norm = layer.input_layernorm.weight
        for name in ("q_proj", "k_proj", "v_proj"):
            module = getattr(layer.self_attn, name)
            if module.bias is not None:
                raise ValueError("Bias unsupported")
            right(module.weight, norm)
        norm.fill_(1)
        left(layer.self_attn.o_proj.weight)
        norm = layer.post_attention_layernorm.weight
        right(layer.mlp.gate.weight, norm)
        right(layer.mlp.experts.gate_up_proj, norm)
        norm.fill_(1)
        left(layer.mlp.experts.down_proj)
    return model


@torch.inference_mode()
def rotate_qwen_residual(model, source, device="cuda", seed=20260911):
    """Rotate Qwen's residual basis, keeping original weight placement."""
    if model.config.model_type != "qwen2_moe" or model.config.tie_word_embeddings:
        raise ValueError("Expected untied Qwen MoE")
    q = hadamard_matrix(source, model.config.hidden_size, device)
    rng = torch.Generator(device=device).manual_seed(seed)
    q = (torch.randint(0, 2, (q.shape[0],), device=device, generator=rng) * 2 - 1)[:, None] * q

    def right(weight, scale=None):
        value = weight.to(device=device, dtype=torch.float64)
        if scale is not None:
            value = value * scale.to(device=device, dtype=torch.float64)
        weight.copy_((value @ q).to(device=weight.device, dtype=weight.dtype))

    def left(weight):
        value = q.T @ weight.to(device=device, dtype=torch.float64)
        weight.copy_(value.to(device=weight.device, dtype=weight.dtype))

    right(model.model.embed_tokens.weight)
    right(model.lm_head.weight, model.model.norm.weight)
    model.model.norm.weight.fill_(1)
    for i, layer in enumerate(model.model.layers):
        norm = layer.input_layernorm.weight
        # Q/K/V outputs keep their coordinates, so their biases stay unchanged.
        for name in ("q_proj", "k_proj", "v_proj"):
            right(getattr(layer.self_attn, name).weight, norm)
        norm.fill_(1)
        left(layer.self_attn.o_proj.weight)
        if layer.self_attn.o_proj.bias is not None:
            left(layer.self_attn.o_proj.bias)
        norm = layer.post_attention_layernorm.weight
        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            right(mlp.gate.weight, norm)
            right(mlp.shared_expert_gate.weight, norm)
            for weight in mlp.experts.gate_up_proj:
                right(weight, norm)
            for weight in mlp.experts.down_proj:
                left(weight)
            dense = mlp.shared_expert
        else:
            dense = mlp
        for name in ("gate_proj", "up_proj"):
            right(getattr(dense, name).weight, norm)
        left(dense.down_proj.weight)
        norm.fill_(1)
        print("rotated Qwen layer", i, flush=True)
    return model
