from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Queue
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from app.server.runtime import RuntimeRegistry
from app.server.tasks import TaskRegistry


@dataclass(slots = True)
class QueueJob:
    job_id: str
    task_name: str
    image_bytes: bytes
    show_intermediate: bool
    generation_kwargs: dict[str, Any]
    status: str = "queued"
    created_at: str = field(
        default_factory = lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory = lambda: datetime.now(timezone.utc).isoformat()
    )
    result: dict[str, Any] | None = None
    error: str | None = None


class TaskQueue:
    def __init__(self, task_registry: TaskRegistry, runtimes: RuntimeRegistry) -> None:
        self.task_registry = task_registry
        self.runtimes = runtimes
        self._pending: Queue[str] = Queue()
        self._jobs: dict[str, QueueJob] = {}
        self._lock = Lock()
        self._worker = Thread(target = self._worker_loop, daemon = True)
        self._worker.start()

    def submit(
        self,
        task_name: str,
        image_bytes: bytes,
        show_intermediate: bool = True,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> str:
        job_id = str(uuid4())
        job = QueueJob(
            job_id = job_id,
            task_name = task_name,
            image_bytes = image_bytes,
            show_intermediate = show_intermediate,
            generation_kwargs = generation_kwargs or {},
        )
        with self._lock:
            self._jobs[job_id] = job
        self._pending.put(job_id)
        return job_id

    def get_job(self, job_id: str) -> QueueJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_status_payload(self, job_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if job is None:
            return None

        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "task_name": job.task_name,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        if job.error is not None:
            payload["error"] = job.error
        if job.result is not None:
            payload["result"] = job.result
        return payload

    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc).isoformat()

    def _worker_loop(self) -> None:
        while True:
            job_id = self._pending.get()
            try:
                self._run_job(job_id)
            finally:
                self._pending.task_done()

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return

        self._update_job(job_id, status = "running")
        try:
            task_result = self.task_registry.run_task(
                task_name = job.task_name,
                image_bytes = job.image_bytes,
                runtimes = self.runtimes,
                generation_kwargs = job.generation_kwargs,
                show_intermediate = job.show_intermediate,
            )
            result = {
                "task_name": task_result.task_name,
                "output": task_result.output,
                "wrappers": [
                    {
                        "wrapper_name": step.wrapper_name,
                        "prompt_task": step.prompt_task,
                        "model_name": step.model_name,
                        "output": step.output,
                        "raw_output": step.raw_output,
                    }
                    for step in task_result.wrappers
                ],
            }
            self._update_job(job_id, status = "completed", result = result)
        except Exception as error:
            self._update_job(job_id, status = "failed", error = str(error))
