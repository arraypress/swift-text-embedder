# TextEmbedder

On-device **text embeddings** for Swift — a text becomes a 1024-number unit vector in a space
where meaning is distance, in 100+ languages and code. Alibaba's Qwen3-Embedding-0.6B
(Apache 2.0) runs on Core AI; the tokenizer, the padding, the pooling and an on-disk passage
index are Swift. Nothing leaves the machine.

```swift
import TextEmbedder

let embedder = try await TextEmbedder(inFolder: modelFolder)     // the .aimodel beside embed-support/
let query = try await embedder.embed(TextEmbedder.query("what is the capital of France?"))
let doc = try await embedder.embed("Paris is the capital and most populous city of France.")
TextEmbedder.similarity(query, doc)                              // 0.76; unrelated text ≈ 0.1

var index = EmbeddingIndex(model: TextEmbedder.assetName, dimension: TextEmbedder.dimension)
for (i, passage) in Passages.split(text, maxTokens: 512, count: embedder.tokenizer.count).enumerated() {
    try index.append(.init(source: path, passage: i, text: passage, modified: date, size: bytes), embedding: try await embedder.embed(passage))
}
try index.save(to: url)
index.search(query, limit: 10)            // passages, best first
index.searchSources(query)                // one hit per document
```

## What it is exactly

The model is Hugging Face's `Qwen3Model` for `Qwen/Qwen3-Embedding-0.6B` (28 layers, hidden
1024, 596 M parameters), exported at two fixed lengths as entry points `t128` and `t512`: token
ids and an attention mask, left-padded with `<|endoftext|>` (151643) as the model card's example
pads, to the last position's hidden state. The host L2-normalises it. A text takes the shortest
length that holds it; a longer text keeps its first 511 tokens and the end-of-text token.

`QwenTokenizer` is Qwen2's byte-level BPE read from the checkpoint's own `vocab.json` and
`merges.txt`: NFC normalisation, Qwen2's pre-tokenising pattern, GPT-2's byte-to-Unicode map,
merges by rank, end-of-text appended. Queries are embedded as
`Instruct: <instruction>\nQuery: <text>`; documents as they are.

`Passages.split` cuts a document at blank lines into passages of at most N tokens (joining
short paragraphs, splitting over-long ones at sentence ends), and `EmbeddingIndex` keeps
passages and unit vectors in one file — a JSON manifest plus raw float32 rows.

## Measured against upstream

`Tools/export_qwen3_embedding.py` writes the asset, the support files and fixtures — 13 texts
(queries with the instruction, documents, Unicode, code, a 60-sentence document) with their
ids, masks and normalised embeddings at each length — and `swift test` compares.

| Stage | Agreement with `transformers` 5.17 |
|---|---|
| Tokenizer, 13 texts × 2 lengths | ids, masks and token counts identical |
| Embeddings, 13 texts × 2 lengths | 125–126 dB PSNR; cosine ≥ 0.999999 |
| Fixed-length padding vs the unpadded model | cosine 0.9999999, max difference 3e-7 |

On an M3 Max, a short text (≤ 128 tokens) embeds in about 0.05 s and a 512-token passage in
about 0.2 s once the model is loaded; loading the 2.4 GB asset takes a few seconds.

## The model

Not bundled (2.4 GB). Export it from the Hugging Face weights, or download the export:

```sh
uv run Tools/export_qwen3_embedding.py --install
hf download arraypress/embed-qwen3 --local-dir models
```

The command-line tool is [`embed`](https://github.com/arraypress/swift-embed-cli).

## Requirements

- macOS 27+ (Core AI), Apple silicon
- Swift 6

## License

MIT — see [LICENSE](LICENSE). The model is Alibaba's, Apache 2.0.
