# flow_wrapper.py
# Generic wrapper for image-task flow:
# input -> process -> generation -> output.

# type hinting
from __future__ import annotations

# standard library imports
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Protocol

# local imports
from app.core.sys_prompt import SystemPrompts

# Protocols for type hinting and extensibility
class PromptTask(Protocol):

    """
    Protocol for prompt task objects that build chat messages.
    """

    def get_message(self, input: Any) -> list[dict[str, Any]]:
        """
        Build model-ready messages from task input.
        """

# Protocol for model adapters that handle processing and generation.
class ModelAdapter(Protocol):
    
    """
    Protocol for model adapters used by the wrapper.
    """

    model_id: str

    def process(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Transform chat messages into model input tensors/structures.
        """

    def generate(
        self, processed_inputs: dict[str, Any], generation_kwargs: dict[str, Any] | None = None
    ) -> str:
        """
        Generate model output text from processed inputs.
        """

# Structured result returned by the wrapper flow.
@dataclass(slots = True)
class InferenceResult:
    
    """
    Structured result returned by the wrapper flow.
    """

    task_name: str
    messages: list[dict[str, Any]]
    raw_output: str
    output: str

# Registry for model adapters to enable extensibility without modifying
# wrapper code.
class ModelRegistry:
    
    """
    Registry to keep model adapter builders extensible.
    """

    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., ModelAdapter]] = {}

    def register(self, model_key: str, builder: Callable[..., ModelAdapter]) -> None:
        """
        Register a model adapter constructor by key.
        """
        self._builders[model_key] = builder

    def create(self, model_key: str, **kwargs: Any) -> ModelAdapter:
        """
        Create a model adapter from the registered constructor.
        """
        if model_key not in self._builders:
            raise KeyError(f"Unknown model key: {model_key}")
        return self._builders[model_key](**kwargs)

# Example adapter for Qwen VL models, implementing the ModelAdapter protocol.
class QwenVLModelAdapter:
    
    """
    Qwen VL adapter that supports current and future Qwen VL checkpoints.
    """

    def __init__(
        self,
        model_id: str = "data/Qwen/Qwen3-VL-8B-Instruct",
        device_map: str | dict[str, Any] | None = "auto",
        torch_dtype: Any = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self._processor: Any | None = None
        self._model: Any | None = None

    def _ensure_loaded(self) -> None:
        """
        Lazily load processor/model only when needed.
        """
        if self._processor is not None and self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        # Prefer multi-modal auto class when available.
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore

            model_class = AutoModelForImageTextToText
        except ImportError:
            model_class = AutoModelForCausalLM

        # Prompt user
        print(f"\033[1;34m[INFO]\033[0m Loading model and processor for {self.model_id}, Please wait.")

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code = self.trust_remote_code
        )
        self._model = model_class.from_pretrained(
            self.model_id,
            trust_remote_code = self.trust_remote_code,
            device_map = self.device_map,
            torch_dtype = self.torch_dtype,
        )

    def process(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Apply chat template and vision preprocessing for Qwen VL models.
        """
        self._ensure_loaded()
        if self._processor is None or self._model is None:
            raise RuntimeError("Model and processor failed to initialize.")

        from qwen_vl_utils import process_vision_info

        prompt_text = self._processor.apply_chat_template(
            messages, tokenize = False, add_generation_prompt = True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = self._processor(
            text = [prompt_text],
            images = image_inputs,
            videos = video_inputs,
            padding = True,
            return_tensors = "pt",
        )
        model_inputs = model_inputs.to(self._model.device)
        return {"model_inputs": model_inputs}

    def generate(
        self, processed_inputs: dict[str, Any], generation_kwargs: dict[str, Any] | None = None
    ) -> str:
        """
        Run generation and decode only newly generated tokens.
        """
        self._ensure_loaded()
        if self._processor is None or self._model is None:
            raise RuntimeError("Model and processor failed to initialize.")

        model_inputs = processed_inputs["model_inputs"]
        kwargs = {"max_new_tokens": 256}
        if generation_kwargs is not None:
            kwargs.update(generation_kwargs)

        generated_ids = self._model.generate(**model_inputs, **kwargs)
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        decoded = self._processor.batch_decode(
            trimmed_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False
        )
        return decoded[0] if decoded else ""


class VisionFlowWrapper:

    """
    Generic flow wrapper with extensible task and model registration.
    """

    def __init__(
        self,
        model_adapter: ModelAdapter,
        task_registry: dict[str, PromptTask] | None = None,
    ) -> None:
        self.model_adapter = model_adapter
        self.task_registry = task_registry or self._build_default_task_registry()

    @staticmethod
    def _build_default_task_registry() -> dict[str, PromptTask]:
        """
        Auto-register nested classes in SystemPrompts that implement get_message.
        """
        registry: dict[str, PromptTask] = {}
        for name, value in inspect.getmembers(SystemPrompts):
            if inspect.isclass(value) and hasattr(value, "get_message"):
                registry[name] = value()
        return registry

    def register_task(self, task_name: str, task_handler: PromptTask) -> None:
        """
        Register or override a task handler.
        """
        self.task_registry[task_name] = task_handler

    def get_available_tasks(self) -> list[str]:
        """
        Return all known task names.
        """
        return sorted(self.task_registry.keys())

    def build_messages(self, task_name: str, input_payload: Any) -> list[dict[str, Any]]:
        """
        Convert task payload into chat messages.
        """
        task_handler = self.task_registry.get(task_name)
        if task_handler is None:
            raise KeyError(f"Unknown task: {task_name}")
        messages = task_handler.get_message(input_payload)
        return messages

    def process(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Process messages using the selected model adapter.
        """
        return self.model_adapter.process(messages)

    def generate(
        self, processed_inputs: dict[str, Any], generation_kwargs: dict[str, Any] | None = None
    ) -> str:
        """
        Generate output using the selected model adapter.
        """
        return self.model_adapter.generate(processed_inputs, generation_kwargs = generation_kwargs)

    def format_output(self, raw_output: str) -> str:
        """
        Normalize output text before returning to callers.
        """
        return raw_output.strip()

    def run(
        self,
        task_name: str,
        input_payload: Any,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> InferenceResult:
        """
        Execute full flow: input -> process -> generation -> output.
        """
        messages = self.build_messages(task_name = task_name, input_payload = input_payload)
        processed_inputs = self.process(messages = messages)
        raw_output = self.generate(
            processed_inputs = processed_inputs, generation_kwargs = generation_kwargs
        )
        output = self.format_output(raw_output = raw_output)
        return InferenceResult(
            task_name = task_name, messages = messages, raw_output = raw_output, output = output
        )

# Example of building a default model registry with Qwen VL adapter.
def build_default_model_registry() -> ModelRegistry:
    """
    Create default model registry with Qwen VL entry.
    default key for qwen: "qwen_vl"
    """
    registry = ModelRegistry()
    registry.register("qwen_vl", QwenVLModelAdapter)
    return registry


# test code
if __name__ == "__main__":

    # Export cuda visible device
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # HF_HOME and TRANSFORMERS_CACHE
    os.environ["HF_HOME"] = "./cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "./cache/huggingface/transformers"
    # TORCH_HOME for PyTorch models
    os.environ["TORCH_HOME"] = "./cache/torch"

    # Build model registry and create adapter instance.
    model_registry = build_default_model_registry()
    qwen_adapter = model_registry.create(
        "qwen_vl",
        model_id = "data/Qwen/Qwen3-VL-8B-Instruct",
        device_map = "cuda:0",
        torch_dtype = "auto",
        trust_remote_code = True,
    )

    # Create wrapper with the adapter.
    wrapper = VisionFlowWrapper(model_adapter = qwen_adapter)

    # List available tasks.
    print("Available tasks:", wrapper.get_available_tasks())

    # Try 'CaptionGeneration' task
    if "CaptionGeneration" in wrapper.get_available_tasks():

        # Example input
        from PIL import Image
        import io

        # Load and prepare the image input as bytes
        image = Image.open("testing_image/chureito_pagoda.jpg")
        image_input: bytes = io.BytesIO()
        image.save(image_input, format = "JPEG")
        image_input = image_input.getvalue()

        # Run the flow
        result = wrapper.run(task_name = "CaptionGeneration", input_payload = image_input)

        # Print results
        print("Raw output:\n    ", result.raw_output)
        print("Formatted output:\n    ", result.output)
