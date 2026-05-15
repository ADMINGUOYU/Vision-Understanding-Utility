from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots = True)
class ServerConfig:
    repo_root: Path
    sample_dir: Path
    model_path: str
    secondary_model_path: str
    device_map: str
    torch_dtype: str
    web_host: str
    web_port: int
    max_new_tokens: int

    @classmethod
    def from_env(cls) -> "ServerConfig":

        repo_root = Path(__file__).resolve().parents[2]

        sample_dir = repo_root / "testing_image"

        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        os.environ.setdefault("HF_HOME", str(repo_root / "cache" / "huggingface"))
        os.environ.setdefault(
            "TRANSFORMERS_CACHE", str(repo_root / "cache" / "huggingface" / "transformers")
        )
        os.environ.setdefault("TORCH_HOME", str(repo_root / "cache" / "torch"))

        return cls(
            repo_root = repo_root,
            sample_dir = sample_dir,
            model_path = os.getenv("MODEL_PATH", "data/Qwen/Qwen3-VL-8B-Instruct"),
            secondary_model_path = os.getenv(
                "SECONDARY_MODEL_PATH", os.getenv("MODEL_PATH", "data/Qwen/Qwen3-VL-8B-Instruct")
            ),
            device_map = os.getenv("DEVICE_MAP", "auto"),
            torch_dtype = os.getenv("TORCH_DTYPE", "auto"),
            web_host = os.getenv("WEB_HOST", "0.0.0.0"),
            web_port = int(os.getenv("WEB_PORT", "6480")),
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256")),
        )
