# Copyright 2026 The BATS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rebuild the benchmark JSONL files used in the paper from official sources.

Benchmark questions and answers are NOT distributed with this repository:
BrowseComp and BrowseComp-ZH are officially released in encrypted form so that
answers do not leak into web crawls (training-data contamination), and we do
not redistribute HLE either. This script reconstructs the exact evaluation
files locally:

  data/browsecomp/bc1266.jsonl   full BrowseComp (1266 questions)
  data/browsecomp/bc200.jsonl    the 200-question subset used in the paper
  data/hle/hle200.jsonl          the 200-question HLE subset used in the paper
  data/bczh/bczh289.jsonl        BrowseComp-ZH (289 questions)

Usage:
  python scripts/prepare_data.py browsecomp
  python scripts/prepare_data.py hle
  python scripts/prepare_data.py bczh --input /path/to/decrypted_browsecomp_zh.jsonl

Run from the repository root. See the "Datasets" section of README.md for
details on the decryption scheme and on obtaining HLE / BrowseComp-ZH.
"""
import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import urllib.request

BROWSECOMP_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
BROWSECOMP_CSV_PATH = "data/browsecomp/test/browse_comp_test_set.csv"


# --- Decryption (same scheme as OpenAI simple-evals) ---
def derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()


def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} records to {path}")


# --- BrowseComp: fully automated (download + decrypt + subset) ---
def prepare_browsecomp():
    if not os.path.exists(BROWSECOMP_CSV_PATH):
        os.makedirs(os.path.dirname(BROWSECOMP_CSV_PATH), exist_ok=True)
        print(f"Downloading official BrowseComp test set from {BROWSECOMP_CSV_URL} ...")
        urllib.request.urlretrieve(BROWSECOMP_CSV_URL, BROWSECOMP_CSV_PATH)
    else:
        print(f"Using existing {BROWSECOMP_CSV_PATH}")

    rows = []
    with open(BROWSECOMP_CSV_PATH, newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            rows.append({
                "uid": idx,
                "problem": decrypt(row["problem"], row["canary"]),
                "answer": decrypt(row["answer"], row["canary"]),
            })
    write_jsonl("data/browsecomp/bc1266.jsonl", rows)

    with open("data/browsecomp/bc200_uids.json") as f:
        subset_uids = json.load(f)
    by_uid = {r["uid"]: r for r in rows}
    subset = [dict(by_uid[u], **{"uid-subset": i}) for i, u in enumerate(subset_uids)]
    write_jsonl("data/browsecomp/bc200.jsonl", subset)


# --- Manifest matching (HLE / BrowseComp-ZH): select our subset by question hash ---
def match_manifest(manifest_path: str, source_rows, out_path: str, extra_fields=()):
    """source_rows: iterable of dicts with 'problem' and 'answer' (plaintext)."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    by_hash = {}
    for r in source_rows:
        by_hash[sha256_text(r["problem"])] = r

    out, missing, answer_mismatch = [], [], 0
    for m in manifest:
        src = by_hash.get(m["problem_sha256"])
        if src is None:
            missing.append(m["uid"])
            continue
        if sha256_text(src["answer"]) != m["answer_sha256"]:
            answer_mismatch += 1
        rec = {"uid": m["uid"], "problem": src["problem"].strip(), "answer": src["answer"].strip()}
        for field in extra_fields:
            if field in m:
                rec[field] = m[field]
        out.append(rec)

    out.sort(key=lambda r: r["uid"])
    write_jsonl(out_path, out)
    if answer_mismatch:
        print(f"WARNING: {answer_mismatch} matched questions have a different answer text than "
              f"the version used in the paper (upstream dataset may have been revised).")
    if missing:
        print(f"WARNING: {len(missing)} of {len(manifest)} questions were not found in the source "
              f"(uids: {missing[:10]}{'...' if len(missing) > 10 else ''}). "
              f"Check that you are using the same dataset version.")
    else:
        print("All manifest entries matched the source dataset.")


def iter_source_file(path: str):
    """Load records from a JSONL or CSV file, normalizing field names.

    Accepts 'problem' or 'question' for the question text. If the file has a
    'canary' column (encrypted BrowseComp-style release), fields are decrypted.
    """
    def normalize(raw):
        problem = raw.get("problem", raw.get("question", ""))
        answer = raw.get("answer", "")
        canary = raw.get("canary", "")
        if canary:
            problem = decrypt(problem, canary)
            answer = decrypt(answer, canary)
        return {"problem": problem, "answer": answer}

    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                yield normalize(raw)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield normalize(json.loads(line))


def prepare_hle(input_path: str = ""):
    if input_path:
        source = list(iter_source_file(input_path))
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("The 'datasets' package is required: pip install datasets")
        print("Loading cais/hle from Hugging Face (gated: accept the terms on the dataset page "
              "and run `huggingface-cli login` first) ...")
        ds = load_dataset("cais/hle", split="test")
        source = [{"problem": r["question"], "answer": r["answer"]} for r in ds]
    match_manifest("data/hle/hle200_manifest.json", source, "data/hle/hle200.jsonl")


def prepare_bczh(input_path: str):
    if not input_path:
        sys.exit("BrowseComp-ZH requires --input: obtain the dataset from the official "
                 "BrowseComp-ZH repository (https://github.com/PALIN2018/BrowseComp-ZH), "
                 "then pass the decrypted (or canary-encrypted) JSONL/CSV file here.")
    source = list(iter_source_file(input_path))
    match_manifest("data/bczh/bczh289_manifest.json", source, "data/bczh/bczh289.jsonl",
                   extra_fields=("topic",))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["browsecomp", "hle", "bczh", "all"])
    p.add_argument("--input", default="", help="Local source file (JSONL/CSV) for hle/bczh.")
    args = p.parse_args()

    if not os.path.isdir("data"):
        sys.exit("Run this script from the repository root (data/ not found).")

    if args.dataset in ("browsecomp", "all"):
        prepare_browsecomp()
    if args.dataset in ("hle", "all"):
        prepare_hle(args.input if args.dataset == "hle" else "")
    if args.dataset in ("bczh", "all"):
        prepare_bczh(args.input if args.dataset == "bczh" else "")


if __name__ == "__main__":
    main()
