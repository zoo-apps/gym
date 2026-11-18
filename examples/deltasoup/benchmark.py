"""Reproduce an independent-writer evaluation of a five-client BitDelta merge."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from bitdelta import merge, pack, tensor_map_digest


FILES = {
    "optdigits.tra": "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd",
    "optdigits.tes": "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8",
}
SCHEMA = "v1:row-major-8x8/count-div-16,tanh32,bias-last-row,digit=0123456789,parity=even-odd"


def load_data(archive, name):
    raw = archive.read(name)
    if hashlib.sha256(raw).hexdigest() != FILES[name]:
        raise ValueError(f"Dataset digest differs: {name}")
    data = np.loadtxt(io.BytesIO(raw), dtype=np.int32, delimiter=",")
    pixels = data[:, :-1].astype(np.float32) / 16
    return with_bias(pixels), data[:, -1]


def with_bias(x):
    return np.column_stack((x, np.ones(len(x), dtype=np.float32)))


def initial_model(rng):
    return {
        name: (rng.normal(0, 0.1, size=shape)).astype(np.float32)
        for name, shape in (("hidden", (65, 32)), ("digit", (33, 10)), ("parity", (33, 2)))
    }


def train(start, x, y, indices, epochs, rng):
    model = {k: v.copy() for k, v in start.items()}
    for _ in range(epochs):
        order = rng.permutation(indices)
        for offset in range(0, len(order), 32):
            ids = order[offset:offset + 32]
            hidden = np.tanh(x[ids] @ model["hidden"])
            features = with_bias(hidden)
            backward = np.zeros_like(hidden)
            for head, labels in (("digit", y[ids]), ("parity", y[ids] % 2)):
                logits = features @ model[head]
                error = np.exp(logits - logits.max(axis=1, keepdims=True))
                error /= error.sum(axis=1, keepdims=True)
                error[np.arange(len(ids)), labels] -= 1
                error *= np.float32(0.5 / len(ids))
                backward += error @ model[head][:-1].T
                model[head] -= np.float32(0.18) * (features.T @ error)
            model["hidden"] -= np.float32(0.18) * (x[ids].T @ (backward * (1 - hidden**2)))
    return model


def accuracy_counts(model, x, y):
    features = with_bias(np.tanh(x @ model["hidden"]))
    return {head: int(np.sum((features @ model[head]).argmax(axis=1) == labels))
            for head, labels in (("digit", y), ("parity", y % 2))}


def run_seed(seed, train_data, test_data):
    x, y = train_data
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    base = train(initial_model(rng), x, y, order[:800], 12, rng)
    shards = np.array_split(order[800:], 5)
    workers = [train(base, x, y, ids, 10, np.random.default_rng(seed * 10 + n))
               for n, ids in enumerate(shards)]
    adapters = [pack(base, worker, SCHEMA) for worker in workers]
    merged = merge(base, adapters, SCHEMA, trim_fraction=0.2)
    full_mean = {k: base[k] + np.mean([w[k] - base[k] for w in workers], axis=0)
                 for k in base}
    no_trim = merge(base, adapters, SCHEMA, trim_fraction=0)
    controls = {"base": base, "full_mean": full_mean, "one_bit_mean": no_trim,
                "one_bit_trimmed": merged}
    return {
        "seed": seed, "train_sizes": [len(ids) for ids in shards],
        "parent": tensor_map_digest(base, SCHEMA), "adapters": [a.digest() for a in adapters],
        "merged": tensor_map_digest(merged, SCHEMA),
        "test_rows": len(test_data[1]),
        "correct": {name: accuracy_counts(model, *test_data) for name, model in controls.items()},
        "individual_correct": [accuracy_counts(worker, *test_data) for worker in workers],
        "full_delta_bytes": sum(w.nbytes for w in base.values()),
        "packed_delta_bytes": sum(len(t.signs) + 4 for t in adapters[0].tensors.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path, help="Unmodified UCI Optical Digits archive")
    args = parser.parse_args()
    with ZipFile(args.zip) as archive:
        training = load_data(archive, "optdigits.tra")
        testing = load_data(archive, "optdigits.tes")
    result = [run_seed(seed, training, testing) for seed in (11, 29, 47)]
    print(json.dumps({"schema": SCHEMA, "numpy": np.__version__, "runs": result}, indent=2))


if __name__ == "__main__":
    main()
