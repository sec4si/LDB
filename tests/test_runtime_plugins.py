from types import SimpleNamespace

import ldb.adapters.sglang as sglang_adapter
import ldb.adapters.vllm as vllm_adapter
from ldb.context import current_batch_context


def test_runtime_plugins_are_dormant_without_explicit_enable(monkeypatch) -> None:
    monkeypatch.delenv("LDB_ENABLE", raising=False)
    sglang_adapter.register_plugin()
    vllm_adapter.register_plugin()


def test_sglang_plugin_registers_worker_initialization_hook(monkeypatch) -> None:
    calls = []
    registry = SimpleNamespace(
        HookRegistry=SimpleNamespace(register=lambda *args: calls.append(args)),
        HookType=SimpleNamespace(AFTER="after", AROUND="around"),
    )
    monkeypatch.setenv("LDB_ENABLE", "1")
    monkeypatch.setattr(sglang_adapter.importlib, "import_module", lambda _: registry)

    sglang_adapter.register_plugin()

    assert calls[0][0] == "sglang.srt.managers.tp_worker.TpModelWorker.__init__"
    assert calls[0][2] == "after"
    assert calls[1][0] == "sglang.srt.model_executor.model_runner.ModelRunner.forward"
    assert calls[1][2] == "around"


def test_vllm_plugin_wraps_worker_model_loading(monkeypatch) -> None:
    installed = []

    class Runner:
        model = object()

    class Worker:
        model_runner = Runner()
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(model="test-model"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            ),
        )
        rank = 0
        world_size = 1

        def load_model(self) -> str:
            return "loaded"

    monkeypatch.setenv("LDB_ENABLE", "1")
    monkeypatch.setattr(
        vllm_adapter.importlib,
        "import_module",
        lambda _: SimpleNamespace(Worker=Worker),
    )
    monkeypatch.setattr(
        vllm_adapter,
        "install_model_runtime",
        lambda *args, **kwargs: installed.append((args, kwargs)),
    )

    vllm_adapter.register_plugin()
    result = Worker().load_model()

    assert result == "loaded"
    assert installed[0][1]["backend"] == "vllm"
    assert installed[0][1]["model_id"] == "test-model"


def test_vllm_runner_context_is_resolved_inside_execute_model(monkeypatch) -> None:
    observed = []

    class GPUModelRunner:
        input_batch = SimpleNamespace(
            req_ids=[],
            req_id_to_index={},
            num_computed_tokens_cpu=[],
            num_prompt_tokens=[],
        )

        def execute_model(self, scheduler_output):
            self.input_batch = SimpleNamespace(
                req_ids=["request-b", "request-a"],
                req_id_to_index={"request-b": 0, "request-a": 1},
                num_computed_tokens_cpu=[5, 1],
                num_prompt_tokens=[5, 4],
            )
            observed.append(current_batch_context())
            return "complete"

    monkeypatch.setattr(
        vllm_adapter.importlib,
        "import_module",
        lambda _: SimpleNamespace(GPUModelRunner=GPUModelRunner),
    )
    vllm_adapter._wrap_model_runner("fake.runner")
    output = GPUModelRunner().execute_model(
        SimpleNamespace(num_scheduled_tokens={"request-a": 3, "request-b": 1})
    )

    assert output == "complete"
    assert [span.request_id for span in observed[0].requests] == ["request-b", "request-a"]
    assert current_batch_context() is None
