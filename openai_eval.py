"""Evaluator utilities that mirror the notebook's judge workflow."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from openai import OpenAI

RESPONSES_FILENAME = "responses.jsonl"
EVAL_RECORDS_FILENAME = "evaluation_records.jsonl"
EVAL_SUMMARY_FILENAME = "evaluation_summary.json"


@dataclass
class EvaluationConfig:
    """Holds evaluator settings that can be tuned per run."""

    run_dir: str
    prompt_template: str
    model: str
    api_key: str
    temperature: float = 0.0
    base_url: Optional[str] = None
    max_tokens: Optional[int] = 1024
    worker_count: int = 4
    seed: Optional[int] = None


class AbstractEvaluator(ABC):
    """Provides the orchestration logic shared across evaluator backends."""

    score_pattern = re.compile(r"score[^0-9]*(\d+(?:\.\d+)?)", re.IGNORECASE)

    def __init__(self, config: EvaluationConfig) -> None:
        if config.worker_count < 1:
            raise ValueError("worker_count must be >= 1")

        self.config = config
        self.run_dir = Path(config.run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {self.run_dir}")

        self._responses_path = self.run_dir / RESPONSES_FILENAME
        self._records_path = self.run_dir / EVAL_RECORDS_FILENAME
        self._summary_path = self.run_dir / EVAL_SUMMARY_FILENAME
        self.run_id = self.run_dir.name
        self._records = self._load_responses()

    def _load_responses(self) -> List[Dict[str, Any]]:
        if not self._responses_path.exists():
            raise FileNotFoundError(
                f"Responses file not found in run folder: {self._responses_path}"
            )

        records: List[Dict[str, Any]] = []
        with self._responses_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "raw_sample" not in payload:
                    raise KeyError(
                        "Each response entry must include the original 'raw_sample' data."
                    )
                records.append(payload)
        return records

    def _format_prompt(self, record: Dict[str, Any]) -> str:
        raw_sample: Dict[str, Any] = record.get("raw_sample", {})
        checklist_items = raw_sample.get("checklist", [])
        checklist_block = ""
        if checklist_items:
            checklist_block = "- " + "\n- ".join(checklist_items)

        template_variables = {
            "history": raw_sample.get("history", ""),
            "user_query": record.get("prompt", ""),
            "prediction": record.get("response", ""),
            "checklist": checklist_block,
        }

        try:
            return self.config.prompt_template.format(**template_variables)
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(
                f"Prompt template references missing variable '{missing}'."
            ) from exc

    def _extract_score(self, response_text: str) -> Optional[float]:
        matches = self.score_pattern.findall(response_text)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except ValueError:  # pragma: no cover - defensive
            return None

    def _persist_records(self, records: List[Dict[str, Any]]) -> None:
        with self._records_path.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _persist_summary(self, summary: Dict[str, Any]) -> None:
        with self._summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    def _summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        overall_scores: List[float] = []
        scores_by_tag: Dict[str, List[float]] = defaultdict(list)

        for row in records:
            score = row.get("score")
            tag = row.get("primary_tag") or "unknown"
            if score is None:
                continue
            overall_scores.append(score)
            scores_by_tag[tag].append(score)

        tag_summary = {
            tag: {
                "count": len(values),
                "average": mean(values) if values else None,
            }
            for tag, values in scores_by_tag.items()
        }

        return {
            "run_id": self.run_id,
            "total_samples": len(records),
            "scored_samples": len(overall_scores),
            "overall_average": mean(overall_scores) if overall_scores else None,
            "per_primary_tag": tag_summary,
        }

    def run(self) -> Dict[str, Any]:
        evaluation_rows: List[Dict[str, Any]] = []
        pending_records = [record for record in self._records if record.get("response")]
        skipped_records = [
            record for record in self._records if not record.get("response")
        ]

        with ThreadPoolExecutor(max_workers=self.config.worker_count) as executor:
            future_map = {
                executor.submit(self._evaluate_single, record): record["index"]
                for record in pending_records
            }
            for future in as_completed(future_map):
                evaluation_rows.append(future.result())

        for skipped in skipped_records:
            evaluation_rows.append(
                {
                    "index": skipped.get("index"),
                    "run_id": self.run_id,
                    "prompt": skipped.get("prompt"),
                    "prediction": skipped.get("response"),
                    "primary_tag": skipped.get("raw_sample", {}).get("primary_tag"),
                    "checklist": skipped.get("raw_sample", {}).get("checklist", []),
                    "prompt_text": None,
                    "evaluation_response": None,
                    "score": None,
                    "error": skipped.get("error")
                    or "No prediction available for evaluation.",
                }
            )

        evaluation_rows.sort(key=lambda row: row["index"])
        summary = self._summarize(evaluation_rows)
        self._persist_records(evaluation_rows)
        self._persist_summary(summary)
        return {
            "records": evaluation_rows,
            "summary": summary,
        }

    def _evaluate_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = self._format_prompt(record)
        response_payload = self._call_model(prompt_text)
        response_text = response_payload.get("response_text") or ""
        score = self._extract_score(response_text)

        row = {
            "index": record.get("index"),
            "run_id": self.run_id,
            "prompt": record.get("prompt"),
            "prediction": record.get("response"),
            "primary_tag": record.get("raw_sample", {}).get("primary_tag"),
            "checklist": record.get("raw_sample", {}).get("checklist", []),
            "prompt_text": prompt_text,
            "evaluation_response": response_text,
            "score": score,
            "error": response_payload.get("error"),
        }
        if response_payload.get("raw_response") is not None:
            row["raw_evaluator_response"] = response_payload["raw_response"]
        return row

    @abstractmethod
    def _call_model(self, prompt_text: str) -> Dict[str, Any]:
        """Send the formatted prompt to the underlying model and return its response."""


class OpenAIEvaluator(AbstractEvaluator):
    """Evaluator implementation that talks to OpenAI-compatible chat endpoints."""

    def __init__(self, config: EvaluationConfig) -> None:
        super().__init__(config)
        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self.client = OpenAI(**client_kwargs)

    def _call_model(self, prompt_text: str) -> Dict[str, Any]:
        messages = [
            {"role": "user", "content": prompt_text},
        ]

        request_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            request_kwargs["max_tokens"] = self.config.max_tokens
        if self.config.seed is not None:
            request_kwargs["seed"] = self.config.seed

        try:
            response = self.client.chat.completions.create(**request_kwargs)
            response_text = response.choices[0].message.content
            raw_response = response.model_dump()
            error: Optional[str] = None
        except Exception as exc:  # pragma: no cover - network/endpoint errors
            response_text = ""
            raw_response = None
            error = str(exc)

        return {
            "response_text": response_text,
            "raw_response": raw_response,
            "error": error,
        }


__all__ = ["EvaluationConfig", "AbstractEvaluator", "OpenAIEvaluator"]
