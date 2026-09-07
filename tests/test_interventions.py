import pytest

from ldb.interventions import apply_tensor_operation
from ldb.models import InterventionKind

torch = pytest.importorskip("torch")


def test_token_scale_changes_only_selected_sequence_position() -> None:
    tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    output = apply_tensor_operation(
        tensor,
        InterventionKind.SCALE,
        value=10.0,
        token=1,
    )

    assert output.tolist() == [[[1.0, 2.0], [30.0, 40.0], [5.0, 6.0]]]


def test_negative_token_index_uses_flattened_live_token_axis() -> None:
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    output = apply_tensor_operation(
        tensor,
        InterventionKind.ABLATE,
        token=-1,
    )

    assert output.tolist() == [[1.0, 2.0], [0.0, 0.0]]


def test_token_selector_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="outside token dimension"):
        apply_tensor_operation(
            torch.ones(1, 2, 3),
            InterventionKind.ABLATE,
            token=2,
        )


def test_semantic_axes_select_attention_query_and_head() -> None:
    tensor = torch.ones(1, 2, 3, 4)

    output = apply_tensor_operation(
        tensor,
        InterventionKind.HEAD_ABLATE,
        head=1,
        num_heads=2,
        token=2,
        token_axis=2,
        head_axis=1,
    )

    assert output[0, 1, 2].count_nonzero() == 0
    assert output.sum().item() == tensor.numel() - 4
