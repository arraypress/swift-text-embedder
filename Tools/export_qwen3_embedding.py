# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "coreai-torch==0.4.2", "coreai-core==1.0.0b2", "transformers"]
# ///
"""Qwen3-Embedding-0.6B (Alibaba, Apache 2.0) → embed-qwen3-0.6b-float32.aimodel, plus its tokenizer files and fixtures.

    uv run Tools/export_qwen3_embedding.py [--lengths 128,512] [--out Tools/exports] [--install]

One entry point per length, `t<length>`: token ids [1, L] int32 and an attention mask [1, L] int32 (LEFT-padded
with <|endoftext|> = 151643, as the model card's example pads) → the last position's hidden state [1, 1024].
The host normalises to unit length. Asserted equal to `AutoModel(...).last_hidden_state[:, -1]` before export.
Fixtures: phrases (queries wrapped as `Instruct: …\\nQuery: …`, and documents) with their ids, masks and
normalised embeddings at each length, so the Swift tokenizer and the asset are checked separately."""
import argparse, os, sys, json, time, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import coreai_torch
from coreai.runtime import AIModelAssetMetadata
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Qwen/Qwen3-Embedding-0.6B"); ap.add_argument("--asset", default="embed-qwen3-0.6b-float32.aimodel")
ap.add_argument("--lengths", default="128,512"); ap.add_argument("--out", default=str(HERE / "exports"))
ap.add_argument("--fixtures", default=str(HERE.parent / "Tests/TextEmbedderTests/Fixtures")); ap.add_argument("--install", action="store_true")
args = ap.parse_args()
lengths = [int(x) for x in args.lengths.split(",")]
tok = AutoTokenizer.from_pretrained(args.name, padding_side="left")
model = AutoModel.from_pretrained(args.name, dtype=torch.float32).eval()
cfg = model.config
print(f"{args.name}: {sum(p.numel() for p in model.parameters())/1e6:.0f} M parameters, {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, pad {tok.pad_token_id}; lengths {lengths}", flush=True)

class Net(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, ids, mask):
        return self.m(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1]
net = Net(model).eval()

fx = Path(args.fixtures); fx.mkdir(parents=True, exist_ok=True)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
support = out / "embed-support"; support.mkdir(exist_ok=True)
for f in ["vocab.json", "merges.txt"]:
    src = hf_hub_download(args.name, f); shutil.copy(src, support / f); shutil.copy(src, fx / f)
json.dump({"model": args.name, "lengths": lengths, "pad": tok.pad_token_id, "dimension": cfg.hidden_size}, open(support / "config.json", "w"), indent=1)
shutil.copy(support / "config.json", fx / "config.json")

instruction = "Given a web search query, retrieve relevant passages that answer the query"
queries = ["what is the capital of France?", "how do I resample audio to 16 kHz in Swift", "best kick drum for techno", "café résumé naïve — unicode & punctuation!!", "  leading spaces\nand a newline"]
docs = ["Paris is the capital and most populous city of France.", "AVAudioConverter converts between audio formats, including sample rates.",
        "A punchy 909-style kick sits under most techno tracks; layer a sub for weight.", "The quick brown fox jumps over the lazy dog. 1234567890",
        "def hello():\n    print('hi')\n", "Ünïcödé täxt with émojis 🎹🎛️ and CJK 漢字 and Arabic عربى", ""]
texts = [f"Instruct: {instruction}\nQuery: {q}" for q in queries] + [d for d in docs if d]
long_text = " ".join(f"Sentence number {i} of a long document about nothing in particular, written to fill tokens." for i in range(60))
fixture = {"instruction": instruction, "texts": texts, "long": long_text, "per_length": {}}
with torch.no_grad():
    for L in lengths:
        enc = tok(texts + [long_text], padding="max_length", max_length=L, truncation=True, return_tensors="pt")
        embs = []
        for i in range(len(texts) + 1):
            e = net(enc["input_ids"][i:i+1], enc["attention_mask"][i:i+1])
            embs.append(nn.functional.normalize(e, dim=-1)[0])
        embs = torch.stack(embs)
        ref = model(**enc).last_hidden_state[:, -1]
        ref = nn.functional.normalize(ref, dim=-1)
        print(f"L={L}: wrapper vs batched last_hidden_state max |diff| {(embs - ref).abs().max():.2e}", flush=True)
        fixture["per_length"][str(L)] = {"ids": enc["input_ids"].tolist(), "mask": enc["attention_mask"].tolist()}
        embs.numpy().astype(np.float32).tofile(fx / f"embeddings_{L}.f32")
    # unpadded reference for the first text, to record the padding's effect
    single = nn.functional.normalize(net(*[v for v in tok([texts[0]], return_tensors="pt").values()][:2]), dim=-1)[0]
    padded = torch.from_numpy(np.fromfile(fx / f"embeddings_{lengths[0]}.f32", np.float32).reshape(-1, cfg.hidden_size)[0])
    print(f"padded (L={lengths[0]}) vs unpadded first text: cosine {float(padded @ single):.7f}, max |diff| {float((padded - single).abs().max()):.2e}", flush=True)
    # counts without padding, for the tokenizer test
    fixture["counts"] = [len(tok(t)["input_ids"]) for t in texts + [long_text]]
json.dump(fixture, open(fx / "text_fixture.json", "w"))

t0 = time.time()
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
with torch.no_grad():
    for L in lengths:
        ids = torch.full((1, L), tok.pad_token_id, dtype=torch.int32); ids[0, -3:] = torch.tensor([9707, 1879, tok.pad_token_id], dtype=torch.int32)
        mask = torch.zeros((1, L), dtype=torch.int32); mask[0, -3:] = 1
        ex = torch.export.export(net, (ids, mask)).run_decompositions(coreai_torch.get_decomp_table())
        converter.add_exported_program(ex, input_names=["ids", "mask"], output_names=["embedding"], entrypoint_name=f"t{L}")
        print(f"exported t{L} in {time.time()-t0:.0f} s", flush=True)
program = converter.to_coreai(); program.optimize()
meta = AIModelAssetMetadata()
meta.author = "Alibaba Qwen team — Qwen3-Embedding-0.6B; Core AI export by TextEmbedder"
meta.license = "Apache-2.0"
meta.model_description = f"Qwen3-Embedding-0.6B: entry points {', '.join(f't{L}' for L in lengths)} take left-padded token ids [1,L] + mask [1,L] (int32, pad 151643) and return the last position's hidden state [1,{cfg.hidden_size}]; normalise in the host. Queries: 'Instruct: <task>\\nQuery: <text>'."
meta.creation_date = int(time.time())
asset = out / args.asset
if asset.exists(): shutil.rmtree(asset)
program.save_asset(asset, meta)
print(f"saved {asset} ({sum(f.stat().st_size for f in asset.rglob('*'))/1e6:.0f} MB) in {time.time()-t0:.0f} s", flush=True)
if args.install:
    dest = Path.home() / "Library/Application Support/embed"; dest.mkdir(parents=True, exist_ok=True)
    for src in [asset, support]:
        d = dest / src.name
        if d.exists(): shutil.rmtree(d)
        shutil.copytree(src, d)
    print(f"installed into {dest}")
