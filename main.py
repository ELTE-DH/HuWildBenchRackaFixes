"""Entry-point orchestrating generation and evaluation stages."""

from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv

from openai_eval import EvaluationConfig, OpenAIEvaluator
from vllm_inference_client import RunnerConfig, VLLMRunner
from data.Wildbench_eval_prompt_template import score_prompt

load_dotenv()


# --------------------
# Editable configuration
# --------------------
GENERATION_CONFIG: Dict[str, object] = {
    "dataset_path": os.getenv("HUWILDBENCH_DATASET_PATH"),
    "prompt_key": "turn[0].content",
    "endpoint_url": os.getenv("VLLM_ENDPOINT_URL"),
    "api_key": os.getenv("VLLM_API_KEY"),
    "model": "elte-nlp/Racka-4B",
    "max_tokens": 8192,
    "temperature": 0.0,
    "call_kwargs": {"frequency_penalty": 0.3, "presence_penalty": 0.3},
    "num_workers": 128,
    "output_root": os.getenv("HUWILDBENCH_OUTPUT_ROOT"),
    "system_prompt": "You are a helpful Hungarian assistant.",
    "normalize_prompts": True,
    "max_samples": None,  # Set to an integer to limit processed prompts
    "use_completions": False,  # Set to True for base models (uses completions API instead of chat)
    "completions_suffix": "\n\nA válasz:",  # Suffix appended to prompts when use_completions=True
}

EVALUATION_CONFIG: Dict[str, object] = {
    "run_dir": "",  # Filled in automatically after generation
    "prompt_template": score_prompt,
    "model": "gpt-4o-2024-08-06",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.0,
    "base_url": None,
    "max_tokens": 8192,
    "worker_count": 32,
    "seed": 42,
}


def run_generation(config: RunnerConfig) -> str:
    runner = VLLMRunner(config)
    runner.run()
    return str(runner.run_dir)


def run_evaluation(config: EvaluationConfig) -> None:
    evaluator = OpenAIEvaluator(config)
    evaluator.run()


def main() -> None:
    generation_cfg = RunnerConfig(**GENERATION_CONFIG)
    run_dir = run_generation(generation_cfg)

    eval_config_dict = dict(EVALUATION_CONFIG)
    eval_config_dict["run_dir"] = run_dir
    eval_cfg = EvaluationConfig(**eval_config_dict)
    run_evaluation(eval_cfg)


if __name__ == "__main__":
    main()
