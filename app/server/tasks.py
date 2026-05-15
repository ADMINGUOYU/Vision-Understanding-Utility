from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from app.server.runtime import RuntimeRegistry


@dataclass(slots = True)
class WrapperResult:
    wrapper_name: str
    prompt_task: str
    model_name: str
    output: str
    raw_output: str


@dataclass(slots = True)
class TaskContext:
    image_bytes: bytes
    generation_kwargs: dict[str, Any]
    show_intermediate: bool
    steps: list[WrapperResult] = field(default_factory = list)

    def add_step(self, step: WrapperResult) -> None:
        self.steps.append(step)

    def get_output(self, wrapper_name: str) -> str:
        for step in reversed(self.steps):
            if step.wrapper_name == wrapper_name:
                return step.output
        raise KeyError(f"Missing wrapper output for {wrapper_name}")


class WrapperRunner(Protocol):
    
    wrapper_name: str

    def run(self, context: TaskContext, runtimes: RuntimeRegistry) -> WrapperResult:
        ...


# PromptWrapper:
# A simple wrapper that runs a single prompt task on a specified model.
# It makes sure the input and output are properly handled and stored in
# the TaskContext.
@dataclass(slots = True)
class PromptWrapper:
    
    wrapper_name: str
    prompt_task: str
    model_name: str = "primary"
    input_builder: Callable[[TaskContext], Any] | None = None

    def run(self, context: TaskContext, runtimes: RuntimeRegistry) -> WrapperResult:
        runtime = runtimes.get(self.model_name)
        input_payload = context.image_bytes if self.input_builder is None else self.input_builder(context)
        inference = runtime.run_prompt(
            task_name = self.prompt_task,
            input_payload = input_payload,
            generation_kwargs = context.generation_kwargs,
        )
        step = WrapperResult(
            wrapper_name = self.wrapper_name,
            prompt_task = self.prompt_task,
            model_name = self.model_name,
            output = inference.output,
            raw_output = inference.raw_output,
        )
        context.add_step(step)
        if context.show_intermediate:
            print(
                f"[TASK] wrapper = {self.wrapper_name} | model = {self.model_name} | output = {step.output}",
                flush = True,
            )
        return step


@dataclass(slots = True)
class TaskResult:
    task_name: str
    output: str
    wrappers: list[WrapperResult]


@dataclass(slots = True)
class TaskDefinition:

    task_name: str
    wrappers: list[WrapperRunner]
    output_builder: Callable[[TaskContext], str] | None = None

    def run(
        self,
        image_bytes: bytes,
        runtimes: RuntimeRegistry,
        generation_kwargs: dict[str, Any] | None = None,
        show_intermediate: bool = True,
    ) -> TaskResult:
        context = TaskContext(
            image_bytes = image_bytes,
            generation_kwargs = generation_kwargs or {},
            show_intermediate = show_intermediate,
        )
        for wrapper in self.wrappers:
            wrapper.run(context, runtimes)
        final_output = (
            context.steps[-1].output
            if self.output_builder is None
            else self.output_builder(context)
        )
        return TaskResult(task_name = self.task_name, output = final_output, wrappers = context.steps)


class TaskRegistry:

    def __init__(self) -> None:
        self._tasks: dict[str, TaskDefinition] = {}

    def register(self, task: TaskDefinition) -> None:
        self._tasks[task.task_name] = task

    def list_tasks(self) -> list[str]:
        return sorted(self._tasks.keys())

    def run_task(
        self,
        task_name: str,
        image_bytes: bytes,
        runtimes: RuntimeRegistry,
        generation_kwargs: dict[str, Any] | None = None,
        show_intermediate: bool = True,
    ) -> TaskResult:
        task = self._tasks.get(task_name)
        if task is None:
            raise KeyError(f"Unknown task: {task_name}")
        return task.run(
            image_bytes = image_bytes,
            runtimes = runtimes,
            generation_kwargs = generation_kwargs,
            show_intermediate = show_intermediate,
        )


# This function builds the default task registry with some example tasks
# and wrappers.
def build_default_task_registry() -> TaskRegistry:
    
    registry = TaskRegistry()

    registry.register(
        TaskDefinition(
            task_name = "CaptionGeneration",
            wrappers = [
                PromptWrapper(
                    wrapper_name = "caption_generation",
                    prompt_task = "CaptionGeneration",
                    model_name = "primary",
                )
            ],
        )
    )

    registry.register(
        TaskDefinition(
            task_name = "PrimaryObjectDetection",
            wrappers = [
                PromptWrapper(
                    wrapper_name = "primary_object_detection",
                    prompt_task = "PrimaryObjectDetection",
                    model_name = "primary",
                )
            ],
        )
    )

    registry.register(
        TaskDefinition(
            task_name = "PrimaryConceptGeneration",
            wrappers = [
                PromptWrapper(
                    wrapper_name = "primary_object_detection",
                    prompt_task = "PrimaryObjectDetection",
                    model_name = "primary",
                ),
                PromptWrapper(
                    wrapper_name = "primary_concept_generation",
                    prompt_task = "PrimaryConceptGeneration",
                    model_name = "secondary",
                    input_builder = lambda context: (
                        context.image_bytes,
                        context.get_output("primary_object_detection"),
                    ),
                ),
            ],
        )
    )

    return registry
