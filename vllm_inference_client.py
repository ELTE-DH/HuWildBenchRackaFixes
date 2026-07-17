"""Configurable threaded runner for vLLM-style OpenAI endpoints."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from dataloading import PromptSample, load_prompt_samples


@dataclass
class RunnerConfig:
    """Holds all tunable parameters for the threaded inference run."""

    dataset_path: str
    prompt_key: str
    endpoint_url: str
    api_key: str
    model: str
    max_tokens: Optional[int] = 512
    temperature: float = 0.0
    call_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_workers: int = 4
    output_root: str = "runs"
    system_prompt: Optional[str] = "You are a helpful Hungarian assistant."
    normalize_prompts: bool = True
    max_samples: Optional[int] = None
    use_completions: bool = False
    completions_suffix: str = "\n\nA válasz:"


class VLLMRunner:
    """Loads prompts, runs inference in parallel, and persists artifacts."""

    def __init__(self, config: RunnerConfig) -> None:
        if config.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.config = config
        self.samples: List[PromptSample] = load_prompt_samples(
            config.dataset_path,
            config.prompt_key,
            normalize=config.normalize_prompts,
        )
        if config.max_samples is not None:
            if config.max_samples <= 0:
                raise ValueError("max_samples must be positive when provided")
            self.samples = self.samples[: config.max_samples]
        self.client = OpenAI(base_url=config.endpoint_url, api_key=config.api_key)
        self.run_dir = self._create_run_dir(Path(config.output_root))
        self._persist_config()
        self._persist_prompts()

    def _create_run_dir(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        candidate = root / f"run_{timestamp}"
        suffix = 1
        while candidate.exists():
            candidate = root / f"run_{timestamp}_{suffix:02d}"
            suffix += 1
        candidate.mkdir()
        return candidate

    def _persist_config(self) -> None:
        config_path = self.run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.config), handle, indent=2, ensure_ascii=False)

    def _persist_prompts(self) -> None:
        prompt_path = self.run_dir / "prompts.jsonl"
        with prompt_path.open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                record = {"index": sample.index, "prompt": sample.prompt}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _call_model(self, sample: PromptSample) -> Dict[str, Any]:
        if self.config.use_completions:
            # Use completions API for base models
            prompt = sample.prompt + self.config.completions_suffix
            request_kwargs: Dict[str, Any] = {
                "model": self.config.model,
                "prompt": prompt,
                "temperature": self.config.temperature,
            }
            if self.config.max_tokens is not None:
                request_kwargs["max_tokens"] = self.config.max_tokens
            request_kwargs.update(self.config.call_kwargs)

            try:
                response = self.client.completions.create(**request_kwargs)
                content = response.choices[0].text
                error: Optional[str] = None
            except Exception as exc:  # pragma: no cover - network/endpoint errors
                content = None
                error = str(exc)
        else:
            # Use chat completions API for instruct models
            messages = []
            if self.config.system_prompt:
                messages.append(
                    {"role": "system", "content": self.config.system_prompt}
                )
            messages.append({"role": "user", "content": sample.prompt})

            request_kwargs: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            if self.config.max_tokens is not None:
                request_kwargs["max_tokens"] = self.config.max_tokens
            request_kwargs.update(self.config.call_kwargs)

            try:
                response = self.client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content
                error: Optional[str] = None
            except Exception as exc:  # pragma: no cover - network/endpoint errors
                content = None
                error = str(exc)

        return {
            "index": sample.index,
            "prompt": sample.prompt,
            "response": content,
            "raw_sample": sample.raw,
            "error": error,
        }

    def _persist_results(self, results: List[Dict[str, Any]]) -> None:
        results = sorted(results, key=lambda item: item["index"])
        outputs_path = self.run_dir / "responses.jsonl"
        with outputs_path.open("w", encoding="utf-8") as handle:
            for record in results:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_map = {
                executor.submit(self._call_model, sample): sample
                for sample in self.samples
            }
            for future in as_completed(future_map):
                results.append(future.result())

        self._persist_results(results)
        return results


__all__ = ["RunnerConfig", "VLLMRunner"]
