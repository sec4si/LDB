from types import SimpleNamespace

from ldb.context import context_from_sglang


class ExtendMode:
    def is_extend(self) -> bool:
        return True

    def is_decode(self) -> bool:
        return False


class DecodeMode:
    def is_extend(self) -> bool:
        return False

    def is_decode(self) -> bool:
        return True


def test_sglang_prefill_context_uses_flattened_token_spans() -> None:
    context = context_from_sglang(
        SimpleNamespace(
            rids=["request-a", "request-b"],
            forward_mode=ExtendMode(),
            extend_seq_lens_cpu=[2, 3],
        )
    )
    assert context.phase == "prefill"
    assert [(item.request_id, item.start, item.end) for item in context.requests] == [
        ("request-a", 0, 2),
        ("request-b", 2, 5),
    ]


def test_sglang_decode_context_assigns_one_row_per_request() -> None:
    context = context_from_sglang(
        SimpleNamespace(rids=["request-a", "request-b"], forward_mode=DecodeMode())
    )
    assert context.phase == "decode"
    assert [(item.start, item.end) for item in context.requests] == [(0, 1), (1, 2)]


def test_vllm_context_uses_final_input_batch_order() -> None:
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["request-b", "request-a"],
            req_id_to_index={"request-b": 0, "request-a": 1},
            num_computed_tokens_cpu=[8, 2],
            num_prompt_tokens=[8, 5],
        )
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"request-a": 3, "request-b": 1})

    from ldb.context import context_from_vllm

    context = context_from_vllm(runner, scheduler_output)

    assert context.phase == "mixed"
    assert [(item.request_id, item.start, item.end) for item in context.requests] == [
        ("request-b", 0, 1),
        ("request-a", 1, 4),
    ]
