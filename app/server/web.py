from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from app.server.config import ServerConfig
from app.server.queue import TaskQueue
from app.server.runtime import RuntimeRegistry
from app.server.tasks import build_default_task_registry


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _is_sample_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS


def _list_samples(sample_dir: Path) -> list[str]:
    if not sample_dir.exists():
        return []
    return sorted([file.name for file in sample_dir.iterdir() if _is_sample_image(file)])


def _load_image_from_request(config: ServerConfig) -> bytes:
    
    selected_sample = request.form.get("sample_name", "").strip()
    uploaded = request.files.get("image_file")

    # Priority to uploaded file if both are provided
    if uploaded is not None and uploaded.filename:
        image_bytes = uploaded.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")
        return image_bytes

    if selected_sample:
        sample_path = (config.sample_dir / selected_sample).resolve()
        if config.sample_dir.resolve() not in sample_path.parents:
            raise ValueError("Invalid sample path.")
        if not _is_sample_image(sample_path):
            raise ValueError("Invalid sample image.")
        return sample_path.read_bytes()

    raise ValueError("Please upload an image or choose one sample image.")


def create_app(config: ServerConfig | None = None) -> Flask:

    app_config = config or ServerConfig.from_env()
    runtimes = RuntimeRegistry.from_config(app_config)
    task_registry = build_default_task_registry()
    task_queue = TaskQueue(task_registry = task_registry, runtimes = runtimes)

    app = Flask(
        __name__,
        template_folder = "templates",
        static_folder = "static",
    )

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            tasks = task_registry.list_tasks(),
            samples = _list_samples(app_config.sample_dir),
        )

    @app.get("/samples/<path:file_name>")
    def sample_image(file_name: str):
        return send_from_directory(app_config.sample_dir, file_name)

    @app.get("/api/tasks")
    def list_tasks():
        return jsonify({"tasks": task_registry.list_tasks()})

    @app.post("/api/submit")
    def submit_task():
        task_name = request.form.get("task_name", "").strip()
        show_intermediate = request.form.get("show_intermediate") == "on"
        
        if task_name not in task_registry.list_tasks():
            return jsonify({"error": f"Unknown task: {task_name}"}), 400

        try:
            image_bytes = _load_image_from_request(app_config)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        job_id = task_queue.submit(
            task_name = task_name,
            image_bytes = image_bytes,
            show_intermediate = show_intermediate,
            generation_kwargs = {"max_new_tokens": app_config.max_new_tokens},
        )
        return jsonify({"job_id": job_id, "status": "queued"})

    @app.get("/api/task/<job_id>")
    def get_task_status(job_id: str):
        payload = task_queue.get_status_payload(job_id)
        if payload is None:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(payload)

    return app


# Program entry
if __name__ == "__main__":
    config = ServerConfig.from_env()
    app = create_app(config)
    app.run(host = config.web_host, port = config.web_port)
