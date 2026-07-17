# HuWildBench Racka Paper Evaluation Fixup

We use this repository to generate and evaluate Hungarian responses for the HuWildBench benchmark while evaluating Racka-4B and baseline models. The evaluation pipeline is designed to be compatible with OpenAI's API, allowing for easy integration with vLLM or other OpenAI-compatible inference endpoints.

The pipeline:

1. Loads prompts from a HuWildBench-style JSONL dataset.
2. Fixes prompt formatting issues and model parametrization.
3. Sends prompts to a local or remote vLLM server.
4. Writes generation artifacts to a timestamped run directory.
5. Evaluates generated responses with the configured OpenAI model.

## Requirements

- Python 3.10 or later
- Access to an OpenAI-compatible inference endpoint, such as vLLM
- An OpenAI API key for the evaluation stage
- A HuWildBench-style JSONL dataset from [OpenHuEval](https://github.com/opendatalab/OpenHuEval)

`vllm` is not a dependency of this client project. The code uses the `openai` Python package to communicate with a separately running vLLM server through its OpenAI-compatible API.

## Setup

Create and activate a virtual environment.

Install the client dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment file from the example:

```bash
cp .env.example .env
```

Edit `.env` and provide real credentials and paths. Do not commit this file.

Place the HuWildBench JSONL dataset in the path specified by `HUWILDBENCH_DATASET_PATH` in `.env`. The dataset can be downloaded from [OpenHuEval](https://github.com/opendatalab/OpenHuEval).

## Environment Variables

| Variable | Purpose | Example |
| --- | --- | --- |
| `HUWILDBENCH_DATASET_PATH` | Path to the input JSONL dataset | `data/HuWildBench.jsonl` |
| `HUWILDBENCH_OUTPUT_ROOT` | Directory for timestamped run artifacts | `runs` |
| `VLLM_ENDPOINT_URL` | Base URL of the OpenAI-compatible inference server | `http://localhost:12344/v1` |
| `VLLM_API_KEY` | API key expected by the inference server | `your-vllm-api-key` |
| `OPENAI_API_KEY` | API key used by the evaluator | `sk-proj-...` |

The tracked [`.env.example`](.env.example) contains masked values. The configuration dictionaries remain in [`main.py`](main.py), where you can adjust model names, token limits, worker counts, prompts, and generation mode.

## Run an Inference Server

Start vLLM independently on a compatible Linux or WSL environment. For example:

```bash
vllm serve elte-nlp/Racka-4B --port 12344 --api-key your-vllm-api-key
```

Ensure `VLLM_ENDPOINT_URL` matches the server URL. The [`main.py`](main.py) file uses the completions API if `use_completions=True` is set. This is the correct setting for raw/base models. Switch it to `False` for chat/instruct/reasoning models.

## Run the Pipeline

With the vLLM server running and `.env` configured:

```powershell
python main.py
```

Generation runs first. Its output directory is passed automatically to the evaluation stage.

## Output

Each run creates `runs/run_YYYYMMDD_HHMMSS/` with:

| File | Contents |
| --- | --- |
| `config.json` | Generation configuration used for the run |
| `prompts.jsonl` | Prepared prompts sent to the inference endpoint |
| `responses.jsonl` | Model responses, source samples, and request errors |
| `evaluation_records.jsonl` | Per-sample judge prompts, results, and scores |
| `evaluation_summary.json` | Aggregate score summary, including primary-tag averages |

The `data/` and `runs/` directories are ignored by Git to prevent committing datasets, generated responses, and evaluation artifacts.


### Reference

This repo is used in our evaluations published as part of the Racka-4B paper. Please use the following citation if you reference this work:

```
@misc{csibi2026rackaefficienthungarianllm,
      title={Racka: Efficient Hungarian LLM Adaptation on Academic Infrastructure}, 
      author={Zsolt Csibi and Bence György Gortka and Natabara Gyöngyössy and Kornél Nagy and Dávid Márk Nemeskey and Martin Sallai and András Simonyi and András Márk Szekeres and Gábor Palkó},
      year={2026},
      eprint={2601.01244},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.01244}, 
}
```

The original HuWildBench dataset's reference is the following:

```
@misc{yang2025openhuevalevaluatinglargelanguage,
      title={OpenHuEval: Evaluating Large Language Model on Hungarian Specifics}, 
      author={Haote Yang and Xingjian Wei and Jiang Wu and Noémi Ligeti-Nagy and Jiaxing Sun and Yinfan Wang and Zijian Győző Yang and Junyuan Gao and Jingchao Wang and Bowen Jiang and Shasha Wang and Nanjun Yu and Zihao Zhang and Shixin Hong and Hongwei Liu and Wei Li and Songyang Zhang and Dahua Lin and Lijun Wu and Gábor Prószéky and Conghui He},
      year={2025},
      eprint={2503.21500},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.21500}, 
}
```

