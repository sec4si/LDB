import pytest

from ldb.graph import GraphTraceSlot

torch = pytest.importorskip("torch")


def test_graph_trace_slot_records_latest_summary_and_occurrences() -> None:
    slot = GraphTraceSlot.create(torch.tensor([1.0, 3.0]))

    slot.observe(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 6.0]))
    slot.observe(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0]))

    count, observed, patched = slot.snapshot()
    assert count == 2
    assert observed.mean == 3.0
    assert observed.std == 1.0
    assert patched.mean == 1.5
    assert patched.norm == pytest.approx(5**0.5)
