from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.flow_wrapper import InferenceResult, VisionFlowWrapper, build_default_model_registry
from app.server.config import ServerConfig

# WrapperRuntime:
# A runtime that wraps a VisionFlowWrapper and exposes a simple interface to run prompts.
@dataclass(slots = True)
class WrapperRuntime:
    model_name: str
    wrapper: VisionFlowWrapper

    def run_prompt(
        self,
        task_name: str,
        input_payload: Any,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> InferenceResult:
        return self.wrapper.run(
            task_name = task_name,
            input_payload = input_payload,
            generation_kwargs = generation_kwargs,
        )

# RuntimeRegistry:
# A registry that holds all the available runtimes (model wrappers) that can be used in
# the task execution. It allows looking up runtimes by their model name and can be
# initialized from a server configuration.
class RuntimeRegistry:
    def __init__(self) -> None:
        self._runtimes: dict[str, WrapperRuntime] = {}

    def register(self, runtime: WrapperRuntime) -> None:
        self._runtimes[runtime.model_name] = runtime

    def get(self, model_name: str) -> WrapperRuntime:
        runtime = self._runtimes.get(model_name)
        if runtime is None:
            raise KeyError(f"Unknown runtime model name: {model_name}")
        return runtime

    @classmethod
    def from_config(cls, config: ServerConfig) -> "RuntimeRegistry":
        registry = cls()
        model_registry = build_default_model_registry()

        primary_adapter = model_registry.create(
            "qwen_vl",
            model_id = config.model_path,
            device_map = config.device_map,
            torch_dtype = config.torch_dtype,
            trust_remote_code = True,
        )
        registry.register(
            WrapperRuntime(
                model_name = "primary",
                wrapper = VisionFlowWrapper(model_adapter = primary_adapter),
            )
        )

        if config.secondary_model_path != config.model_path:
            secondary_adapter = model_registry.create(
                "qwen_vl",
                model_id = config.secondary_model_path,
                device_map = config.device_map,
                torch_dtype = config.torch_dtype,
                trust_remote_code = True,
            )
            registry.register(
                WrapperRuntime(
                    model_name = "secondary",
                    wrapper = VisionFlowWrapper(model_adapter = secondary_adapter),
                )
            )
        else:
            # If the secondary model is the same as the primary,
            # we can just reuse the primary wrapper.
            registry.register(
                WrapperRuntime(
                    model_name = "secondary",
                    wrapper = registry.get("primary").wrapper,
                )
            )

        return registry
