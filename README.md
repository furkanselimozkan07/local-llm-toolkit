# local-llm-toolkit

Python tools for running LLMs locally with [Ollama](https://ollama.com): chat from the command
line, ask questions about your own documents (RAG), test prompts and compare local models.
See [ROADMAP.md](ROADMAP.md) for what is done and what is planned.

*(Türkçe: Kendi belgelerinle soru-cevap yapan bir araç (RAG), komut satırı sohbet, prompt testi ve yerel modelleri karşılaştırma.)*

## Features
- Typed `OllamaClient`: list models, `generate`, `chat`, streaming variants, embeddings
- Tokens-per-second statistics for every non-streamed generation
- Clear `OllamaError` for unreachable servers and API errors
- Fully tested without a model, using an in-process fake Ollama server

## Install
```bash
git clone https://github.com/furkanselimozkan07/local-llm-toolkit
cd local-llm-toolkit
pip install -e ".[dev]"
```

## Usage
```python
from llm_toolkit import ChatMessage, OllamaClient

client = OllamaClient()                      # http://localhost:11434
print(client.list_models())

res = client.generate("llama3", "Explain PID control in one sentence.")
print(res.text, f"({res.tokens_per_second:.1f} tok/s)")

for piece in client.stream_chat("llama3", [ChatMessage("user", "Hello!")]):
    print(piece, end="", flush=True)

vectors = client.embed("nomic-embed-text", ["first text", "second text"])
```

## Tests
```bash
pytest
```

## License
MIT
