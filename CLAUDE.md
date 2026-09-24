# CLAUDE.md — swift-text-embedder

Qwen3-Embedding-0.6B on Core AI: `TextEmbedder` (entry points `t128`/`t512`,
left padding, last-token pooling + L2), `QwenTokenizer` (Qwen2 byte-level
BPE), `EmbeddingIndex` + `Passages`. Module `TextEmbedder`; the CLI is
`../swift-embed-cli` (binary `embed`). Added 2026-09-24.

## Build & test
```bash
swift build && swift test        # tokenizer, index, passages need no model
```
Model-backed tests need `~/Library/Application Support/embed/` with the
asset and `embed-support/`: `uv run Tools/export_qwen3_embedding.py --install`
(probe venv; 2.4 GB asset, export takes ~30 s).

## Traps
- **Special tokens are not in vocab.json.** `<|endoftext|>` (151643) is an
  "added token" in tokenizer.json only; the export writes its id into
  `embed-support/config.json` (`pad`) and the tokenizer reads it there.
- transformers 5: `rope_theta` moved into `config.rope_parameters`;
  `dtype=` not `torch_dtype=`.
- Left padding to a fixed length changes an embedding by ≤ 3e-7 — RoPE is
  relative — so fixed-length entry points are exact enough; keep the mask.
- Passages of an over-long paragraph are final; never re-join them with
  other paragraphs, or the token budget is blown.
- Same GPU rule as the rest: one Core AI job at a time.
