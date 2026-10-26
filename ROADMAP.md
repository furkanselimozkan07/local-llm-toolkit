# Roadmap

## Vision
A small, dependency-light Python toolkit for working with LLMs that run locally through
[Ollama](https://ollama.com): talk to models, ask questions about your own documents (RAG),
test prompts, and compare local models on speed and quality — all offline and testable
without a running model.

## Backlog
- [x] Project layout + typed Ollama HTTP client (generate, chat, streaming, embeddings)
- [ ] Interactive CLI chat (`llm-chat`) with history, system prompt and `/commands`
- [ ] Persist and resume chat sessions as JSON
- [ ] Text chunking utilities (fixed size, overlap, sentence-aware)
- [ ] Document loaders: `.txt`, `.md`, `.pdf`
- [ ] Embedding helper with on-disk cache
- [ ] Cosine-similarity vector store (pure Python/NumPy) with save/load
- [ ] RAG pipeline: index a folder, retrieve top-k, build the prompt
- [ ] `llm-rag` CLI: `index` and `ask` commands with source citations
- [ ] Prompt template engine with variables and few-shot examples
- [ ] Prompt test-suite format (YAML/JSON) with assertions (contains, regex, JSON schema)
- [ ] Prompt test runner CLI with pass/fail report
- [ ] Model benchmark: tokens/s, time to first token, memory notes
- [ ] Model comparison report (Markdown table) across several models
- [ ] LLM-as-judge quality scoring for comparisons
- [ ] Tool/function-calling agent loop with a few built-in tools (calculator, file reader)
- [ ] Structured JSON output helper with validation and retry
- [ ] Conversation memory: sliding window and summarisation
- [ ] Example notebooks / scripts in `examples/`
- [ ] GitHub Actions CI running pytest

## Changelog
- 2026-10-26 — Set up package layout (`src/llm_toolkit`, tests, pyproject, MIT license) and added
  `OllamaClient` with generate/chat/streaming/embeddings and tokens-per-second stats; 8 tests
  against an in-process fake Ollama server.
