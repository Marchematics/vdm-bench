"""Download and lay out the datasets expected by VDM-Bench.

Usage: python scripts/prepare_data.py --root /path/to/data
Layout produced:
  sift/train.npy   (N x 128 float32)   sift/test.npy (10k x 128)
  mnist/train.npy  (60k x 784 float32) mnist/test.npy (10k x 784)
  scifact/train.npy (5,183 x 384)      scifact/test.npy (1,109 x 384)
"""
import argparse
import gzip
import io
import os
import urllib.request

import numpy as np


def sift(root):
    """SIFT1M from the open-vdb parquet mirror (HuggingFace or ModelScope)."""
    import pyarrow.parquet as pq
    os.makedirs(os.path.join(root, "sift"), exist_ok=True)
    base = "https://huggingface.co/datasets/open-vdb/sift-128-euclidean/resolve/main"
    for split, out in [("train/train-00001-of-00001.parquet", "train.npy"),
                       ("test/test-00001-of-00001.parquet", "test.npy")]:
        p = os.path.join(root, "sift", out)
        if os.path.exists(p):
            continue
        url = f"{base}/{split}"
        print("downloading", url)
        urllib.request.urlretrieve(url, p + ".parquet")
        t = pq.read_table(p + ".parquet").column("emb").combine_chunks()
        a = np.asarray(t.values).reshape(len(t), -1).astype(np.float32)
        np.save(p, a)
        os.remove(p + ".parquet")
        print(out, a.shape)


def mnist(root):
    os.makedirs(os.path.join(root, "mnist"), exist_ok=True)
    urls = {
        "train_i": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "test_i": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    }
    for key, url in urls.items():
        raw = urllib.request.urlopen(url, timeout=120).read()
        buf = gzip.GzipFile(fileobj=io.BytesIO(raw))
        magic = int.from_bytes(buf.read(4), "big")
        n = int.from_bytes(buf.read(4), "big")
        r = int.from_bytes(buf.read(4), "big")
        c = int.from_bytes(buf.read(4), "big")
        a = np.frombuffer(buf.read(), dtype=np.uint8).reshape(n, r * c).astype(np.float32)
        out = "train.npy" if key.startswith("train") else "test.npy"
        np.save(os.path.join(root, "mnist", out), a)
        print(out, a.shape)


def scifact(root):
    """Corpus/query embeddings; requires `datasets` and `sentence-transformers`."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    os.makedirs(os.path.join(root, "scifact"), exist_ok=True)
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = model.encode([d["text"] for d in corpus], batch_size=64, show_progress_bar=True)
    Q = model.encode([q["text"] for q in queries], batch_size=64, show_progress_bar=True)
    np.save(os.path.join(root, "scifact", "train.npy"), X.astype(np.float32))
    np.save(os.path.join(root, "scifact", "test.npy"), Q.astype(np.float32))
    print(X.shape, Q.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    if a.only in ("all", "sift"):
        sift(a.root)
    if a.only in ("all", "mnist"):
        mnist(a.root)
    if a.only in ("all", "scifact"):
        scifact(a.root)
