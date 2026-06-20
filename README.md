## Introduction
This repository demonstrates an end-to-end AI evaluation pipeline that automatically scores 
chatbot responses for faithfulness and answer relevancy using LLM-as-a-judge. Responses are 
generated live against a simulated RAG knowledge base and evaluated against a pre-defined 
quality threshold.

**Tech Stack**: Python, DeepEval, OpenAI GPT-4o-mini

## Purpose
Businesses are integrating AI chatbots into customer-facing workflows to automate 
responses at scale. Without automated quality checks, chatbots can return irrelevant 
answers or hallucinate — confidently stating information that contradicts their own 
knowledge base. This pipeline demonstrates how to catch those failures automatically 
before they reach end users.

## How It Works
1. A question is passed to the AusClaim AI chatbot
2. The chatbot generates a live response constrained to the RAG knowledge base
3. DeepEval evaluates the response against two metrics:
   - **Faithfulness** — does the response contradict the knowledge base? (hallucination detection)
   - **Answer Relevancy** — does the response directly address the question asked?
4. Each metric scores 0.0 to 1.0 — responses scoring below 0.7 fail the quality gate

## Prerequisites

### All platforms
- [pyenv](https://github.com/pyenv/pyenv) for Python version management
- An OpenAI API key with billing enabled — [platform.openai.com](https://platform.openai.com)

### Linux/Ubuntu — install build dependencies before pyenv
```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils \
tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### Install pyenv
**Linux/macOS:**
```bash
curl https://pyenv.run | bash
```
Add to `~/.bashrc`:
```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
```

**macOS — install build dependencies via Xcode:**
```bash
xcode-select --install
```

**Windows:**
Use [pyenv-win](https://github.com/pyenv-win/pyenv-win)
> No build dependencies required — pyenv-win downloads pre-compiled Python binaries.

> This project was built and tested on Ubuntu 24.04 LTS.

## How to run it
1. Clone the repo
2. Install Python 3.11.9 — `pyenv install 3.11.9`
3. Navigate into the repo directory and pin the Python version — `pyenv local 3.11.9`
4. Create a virtual environment — `python -m venv venv`
5. Activate the virtual environment - `source venv/bin/activate`
6. Install dependencies — `pip install -r requirements.txt`
7. Create a `.env` file in the root directory and add your OpenAI API key:
```
   OPENAI_API_KEY=your-key-here
```
8. Run the evaluation — `python src/evaluators/ausclaim_eval.py`

## Known Limitations
- **Parallel evaluation timeouts** — DeepEval runs test cases asynchronously by default. 
  On resource-constrained machines this can cause timeouts. Test cases are run sequentially 
  in this implementation as a workaround.
- **Borderline test cases** — Partially correct responses with complex reasoning requirements 
  occasionally cause evaluation timeouts. Root cause is under investigation — likely related 
  to async behaviour on resource-constrained machines or API response latency.