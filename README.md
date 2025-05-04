# Agent Evals

Welcome to the Agent Evals repo.

This repo contains the source code for the paper:  
**"Deliberate Thought in LLMs: A Study of Reasoning Depth Across Tasks and Frameworks"**

The main objective of this project is to evaluate various agentic frameworks with different degrees of multi-step reasoning to understand their impact on performance across tasks.

We evaluate three task types:
- **Mathematical Reasoning** (AIME, GSM8K)
- **Coding** (HumanEval)
- **Creative SVG Design** (Space Scene Generation)

We test across four major agentic frameworks:
- `PydanticAI`
- `LangChain`
- `CrewAI`
- `AutoGen`

And we include two baselines:
- `GPT-4o`
- `o3-mini` (a highly-capable reasoning model)

---

## Setup

Install dependencies using Poetry:

```bash
poetry install
````

Then, set your environment variables:

```bash
export AI_SANDBOX_API_KEY=<your sandbox key>
```

---

To run the evaluations locally, some files give you the option to specify the cli argument `--n`, which is the number of problems to evaluate. Likewise, some allow you to specify `--reasoning`, the number of reasoning steps. This can be 1, 2, or 3.

## Mathematical Reasoning

### AIME

Run structured output (no reasoning; includes o3-mini):

```bash
python src/evaluate/math/aime/eval_aime_structured_output.py --n <n>
```

Run with reasoning (1-step, 2-step, or 3-step):

```bash
python src/evaluate/math/aime/eval_aime_reasoning.py --n <n> --reasoning <r>
```

### GSM8K

Run structured output:

```bash
python src/evaluate/math/aime/eval_gsm8k_structured_output.py --n <n>
```

Run with reasoning:

```bash
python src/evaluate/math/aime/eval_gsm8k_reasoning.py --n <n> --reasoning <r>
```

---

## Code Generation

Evaluate HumanEval (Python coding tasks):

### Structured Output:

```bash
python src/evaluate/math/aime/eval_coding_structured_output.py --n <n>
```

### With Reasoning:

```bash
python src/evaluate/math/aime/eval_coding_reasoning.py --n <n> --reasoning <r>
```

Each run will print out accuracy scores for each framework.

---

## SVG Generation

To evaluate creative SVG rendering of space scenes:

```bash
python src/evaluate/svg/eval_svg_reasoning.py --reasoning <r>
```

* `r` is the number of reasoning steps: 1, 2, or 3.
* The o3-mini model runs when `--reasoning 1` is selected.

Generated outputs are saved to:

```bash
src/svg_outputs/1_step/
src/svg_outputs/2_step/
src/svg_outputs/3_step/
```

Open these `.html` files in a browser to view the generated SVG space scenes.

---

Happy evaluating!
