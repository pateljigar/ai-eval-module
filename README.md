# ai-eval-module

An end-to-end AI evaluation pipeline for a LangGraph claims triage agent. Built to demonstrate how to evaluate multi-step LLM agents — not just their final output, but each reasoning step independently.

**Tech Stack**: Python 3.11, LangGraph, LangChain, DeepEval, ChromaDB, Langfuse, Promptfoo, RAGAS, OpenAI GPT-4o-mini, GitHub Actions

---

## Introduction

This repository demonstrates an end-to-end AI evaluation pipeline that automatically scores chatbot responses for faithfulness and answer relevancy using LLM-as-a-judge. Responses are generated live against a simulated RAG knowledge base and evaluated against a pre-defined quality threshold.

It also evaluates a multi-step LangGraph claims triage agent — checking each reasoning node independently so failures can be traced to the exact step where they occur. Module 9 extends this by evaluating the `research_policy` node at the retrieval and generation level independently, using RAGAS to score whether ChromaDB retrieved the right policy rule and whether the LLM used it faithfully.

---

## Purpose

Businesses are integrating AI chatbots into customer-facing workflows to automate responses at scale. Without automated quality checks, chatbots can return irrelevant answers or hallucinate — confidently stating information that contradicts their own knowledge base. This pipeline demonstrates how to catch those failures automatically before they reach end users.

---

## What this builds

A nine-module portfolio project that goes from a raw LLM call to a fully automated CI evaluation pipeline.

| Module   | What it does                                                                   |
| -------- | ------------------------------------------------------------------------------ |
| Week 0   | DeepEval RAG pipeline — baseline faithfulness and relevancy evaluation         |
| Module 1 | LangGraph claims triage agent — the system under test                          |
| Module 2 | Per-node DeepEval evaluators using GEval                                       |
| Module 3 | Shared agent execution — single run, all test functions                        |
| Module 4 | Multi-claim coverage across 4 claim types, deterministic at `temperature=0`    |
| Module 5 | GitHub Actions CI — evaluator runs automatically on every PR                   |
| Module 6 | ChromaDB vector retrieval — replaces hardcoded policy knowledge base           |
| Module 7 | Langfuse observability — per-node trace, token, and cost tracking              |
| Module 8 | Promptfoo red-teaming — adversarial test suite with prompt injection hardening |
| Module 9 | RAGAS evaluation — retrieval and generation quality scoring for `research_policy` node |

---

## Architecture

```
claim_input
    │
    ▼
[classify_claim]      → claim_type, urgency
    │
    ▼
[research_policy]     → ChromaDB vector retrieval → policy_findings
    │
    ▼
[summarise_decision]  → final_decision { claim_type, urgency, recommendation }
    │
    ▼
[DeepEval GEval]      → per-node scores, pass/fail
    │
    ▼
[RAGAS]               → retrieval and generation quality scores for research_policy
    │
    ▼
[Langfuse]            → traces, token usage, cost per node
    │
    ▼
[Promptfoo]           → adversarial red-teaming evaluation
    │
    ▼
[GitHub Actions]      → CI pipeline on every PR
```

---

## How It Works

**Week 0 RAG pipeline:**

1. A question is passed to the AusClaim AI chatbot
2. The chatbot generates a live response constrained to the RAG knowledge base
3. DeepEval evaluates the response against two metrics:
   - **Faithfulness** — does the response contradict the knowledge base? (hallucination detection)
   - **Answer Relevancy** — does the response directly address the question asked?
4. Each metric scores 0.0 to 1.0 — responses scoring below 0.7 fail the quality gate

**Modules 1–9 LangGraph pipeline:**
1. A claim is passed to the AusClaim AI triage agent
2. The agent classifies the claim, retrieves relevant policy from ChromaDB, and produces a structured decision
3. DeepEval evaluates each node independently using GEval
4. RAGAS evaluates the `research_policy` node separately — scoring retrieval quality (Context Precision, Context Recall) and generation quality (Faithfulness, Answer Relevancy)
5. Langfuse captures every LLM call — prompt, response, tokens, latency, cost
6. Promptfoo runs adversarial test cases against the agent on every PR
7. GitHub Actions runs the full pipeline automatically

---

## Claim types supported

- `motor_vehicle` — police report required if damage exceeds $2500
- `property` — photos and repair quotes within 30 days
- `public_liability` — incident report and witness statements within 14 days
- `other` — manual review by a senior assessor

---

## Evaluation metrics

| Node                 | Metric                        | Threshold |
| -------------------- | ----------------------------- | --------- |
| `classify_claim`     | Claim Classification Accuracy | 0.70      |
| `research_policy`    | Policy Research Accuracy      | 0.70      |
| `summarise_decision` | Decision Summary Accuracy     | 0.60      |

**RAGAS metrics — `research_policy` node:**

| Metric             | What it checks                                              |
| ------------------ | ----------------------------------------------------------- |
| Context Precision  | Did ChromaDB retrieve the right policy rule?                |
| Context Recall     | Did the retrieved rule contain everything needed to answer? |
| Faithfulness       | Did the LLM summary stick to what was retrieved?            |
| Answer Relevancy   | Did the summary directly address the question asked?        |

---

## Documented failure modes

**Silent state corruption** — `research_policy` writes to `claim_type` instead of `policy_findings`. LangGraph completes without error. The final recommendation is wrong. The evaluator catches it by asserting `claim_type` is a valid enum value and `policy_findings` is non-empty.

**Hallucination** — `summarise_decision` asserts damage exceeded $2500 when no amount was stated in the claim. Fixed by adding a conditional language guard to the system prompt.

**LLM non-determinism** — `temperature=0` reduces output variation. GEval semantic scoring handles remaining variation without requiring exact string matching.

**Prompt injection (partial)** — adversarial input attempted to override urgency classification via embedded instructions. The agent was hardened by requiring explicit evidence before escalating urgency. Caught and fixed via Promptfoo red-teaming.

---

## Evaluation findings

**Answer Relevancy is lower for the `other` claim type (0.45 vs 0.77–0.84 for other types).** This is not a bug — it reflects a genuine design characteristic. The `other` policy rule is a process instruction ("escalate to a senior assessor") rather than a requirements list. RAGAS generates reverse-engineered questions from the policy summary to compute relevancy; with no concrete requirements to reconstruct, cosine similarity against the original question stays low. GEval alone would not surface this — RAGAS revealed it by evaluating the retrieval and generation steps independently.

**Answer Relevancy improved significantly when the question was constructed rather than inferred.** Initial runs passed bare `claim_type` (e.g. `"motor_vehicle"`) as `user_input`, producing scores of 0.03–0.54. Replacing it with `"What are the requirements for a {claim_type} insurance claim?"` — which matches the query shape already used internally by `policy_store.query_policy()` — raised scores to 0.45–0.84. This confirmed that RAGAS Answer Relevancy is sensitive to question quality, not just answer quality.

---

## Prerequisites

- An OpenAI API key with billing enabled — [platform.openai.com](https://platform.openai.com)
- A Langfuse account (free cloud tier) — [langfuse.com](https://langfuse.com)
- [pyenv](https://github.com/pyenv/pyenv) for Python version management

> **Platform support:** These setup steps have been tested on Ubuntu 24.04 LTS only.
> For macOS and Windows, refer to the official installation guides:
>
> - pyenv (Linux/macOS): https://github.com/pyenv/pyenv
> - pyenv-win (Windows): https://github.com/pyenv-win/pyenv-win

---

## Setup (Ubuntu 24.04 LTS)

**Step 1 — update package list:**

```bash
sudo apt update
```

**Step 2 — install build dependencies:**

```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils \
tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

**Step 3 — install pyenv:**

```bash
curl https://pyenv.run | bash
```

**Step 4 — add to `~/.bashrc`:**

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

**Step 5 — reload shell:**

```bash
source ~/.bashrc
```

**Step 6 — install Node.js 22 (required for Promptfoo):**
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

Verify:
```bash
node --version
npm --version
```

**Step 7 — apply ragas compatibility patch:**

`ragas==0.4.3` ships with a broken import on `langchain-community>=0.4` installations. Apply this patch after installing dependencies:

```bash
python - << 'EOF'
path = "venv/lib/python3.11/site-packages/ragas/llms/base.py"
with open(path, "r") as f:
    content = f.read()
old = "from langchain_community.chat_models.vertexai import ChatVertexAI\nfrom langchain_community.llms import VertexAI"
new = "from langchain_google_vertexai import ChatVertexAI, VertexAI"
if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Already patched or pattern not found")
EOF
```

See: https://github.com/vibrantlabsai/ragas/issues/2745

---

## How to run it

1. Clone the repo
2. Install Python 3.11.9 — `pyenv install 3.11.9`
3. Navigate into the repo directory and pin the Python version — `pyenv local 3.11.9`
4. Create a virtual environment — `python -m venv venv`
5. Activate the virtual environment — `source venv/bin/activate`
6. Install dependencies — `pip install -r requirements.txt`
7. Apply the ragas compatibility patch (see Setup Step 7 above)
8. Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your-key-here
LANGFUSE_SECRET_KEY=your-langfuse-secret-key-here
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key-here
LANGFUSE_BASE_URL=your-langfuse-base-url-here
```

**Run the Week 0 RAG evaluator:**

```bash
python src/evaluators/ausclaim_eval.py
```

**Run the agent directly:**

```bash
python src/agents/claims_triage_agent.py
```

**Run the claims triage evaluator:**

```bash
python src/evaluators/claims_triage_evaluator.py
```

**Run the RAGAS evaluator:**

```bash
python src/evaluators/ragas_evaluator.py
```

**Run the Promptfoo red-teaming evaluation:**
```bash
npx promptfoo@latest eval -c promptfoo.yaml
```

CI runs automatically on every PR via GitHub Actions.

---

## Known Limitations

- **Parallel evaluation timeouts** — DeepEval runs test cases asynchronously by default. On resource-constrained machines this can cause timeouts. Test cases are run sequentially in this implementation as a workaround.
- **Borderline test cases** — Partially correct responses with complex reasoning requirements occasionally cause evaluation timeouts. Root cause is under investigation — likely related to async behaviour on resource-constrained machines or API response latency.
- **LLM non-determinism** — `temperature=0` reduces but does not eliminate output variation. OpenAI does not guarantee identical outputs even at zero temperature.
- **In-memory vector store** — ChromaDB runs in-memory and reloads policy rules on every agent start. A persistent store is planned as a future improvement.
- **ragas upstream bug** — `ragas==0.4.3` has a broken import on `langchain-community>=0.4` installations. A manual patch is required post-install. See Setup Step 7.

---

> This project was built and tested on Ubuntu 24.04 LTS.