# BitDelta and coordinate-wise trimmed merging on handwritten digits

This CPU experiment trains five independent updates against the same checkpoint, compresses each update, and measures the merged model on new writers. It implements the per-tensor BitDelta encoding and the trimmed-mean stage of the [Zoo BitDelta/DeltaSoup paper](https://github.com/zoo-apps/papers/blob/e12b0a17b6c39bb641d84cfe04be96314b1d199a/zoo-bitdelta-deltasoup/zoo-bitdelta-deltasoup.tex) for [Gym #6](https://github.com/zoo-apps/gym/issues/6).

## Reproduce

Run from this directory. The reference environment used Python 3.14.4 and NumPy 2.5.1. The ZIP stays local. The benchmark verifies the SHA-256 of both data files before training.

```sh
python -m pip install numpy==2.5.1
curl -fL 'https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip' -o optdigits.zip
python -m unittest -v test_bitdelta
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py optdigits.zip > result.json
python -c "import json; assert json.load(open('result.json')) == json.load(open('results.json')); print('Full reference report matched')"
```

## Data and protocol

E. Alpaydin and C. Kaynak's [UCI Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Inputs are 8 × 8 pixel counts divided by 16. One 32-unit tanh layer feeds a ten-class digit head and a separate binary parity head. Each head contributes half of the training loss.

For each seed (11, 29, 47), 800 randomly selected rows from the official training split train the common base for 12 epochs. The remaining 3,023 training rows are partitioned into five disjoint shards of 604 or 605 rows. Each worker starts at the same base and trains both heads for ten epochs on its own shard. All models use the same learning rate, 0.18, and batch size, 32. Every reported score uses the same official 1,797-row test set from 13 writers absent from the training set. The two heads share inputs, so their scores are correlated.

## Adapter and merge specification, version 1

Each update tensor stores its mean absolute change as a float32 scale and one little-bit-order sign per parameter in row-major order. A zero change gets a positive sign. An all-zero tensor has scale zero. The digest of the parent binds the input and class conventions, tensor names, shapes and little-endian float32 bytes. Each adapter digest incorporates that parent and its packed tensor payloads. The report records the base, all five adapter digests, and the merged-model digest for each seed. These digests identify content and do not authenticate a contributor's identity.

The merger verifies the parent and schema, checks payload lengths, finite nonnegative scales and padding bits, and rejects duplicate adapters. For the trimmed merge, each coordinate drops the smallest and largest of five decoded values and averages the remaining three. Controls use the base alone, the uncompressed mean, and the decoded one-bit mean with no trimming. There is no model selection on the test set.

The finite-range tests include two float32 updates of approximately 2 × 10³⁸ and 3 × 10³⁸. Their sum overflows but their mean fits. The merger recomputes affected coordinates in float64 and rejects a reconstructed weight outside float32 range. The reference experiment uses ordinary float32 arithmetic throughout and has the same model digests with this check.

## Results

Mean accuracy over the three seeds, with integer counts for every seed, worker, and control in [results.json](results.json).

| Model | Digit accuracy | Parity accuracy |
| --- | ---: | ---: |
| Common base | 91.02% | 90.58% |
| Full-precision mean | 92.38% | 92.54% |
| One-bit mean | 91.99% | 91.49% |
| One-bit trimmed mean | 91.89% | 91.54% |

The trimmed merge gained 0.87 percentage points on digit recognition and 0.96 on parity over the base. The full-precision mean scored higher on both. Each packed update used 323 bytes for signs and scales versus 9,904 bytes of float32 deltas (30.7×). These sizes exclude manifests, framing and the shared checkpoint.

This run covers honest local workers, two related classification heads and one small architecture. It does not measure language-model adapters, independent-worker authentication, safety-gated greedy soup selection, Byzantine tolerance or end-to-end transfer size. Floating-point differences across BLAS implementations can change model hashes, so comparisons across environments should include counts and dependency versions.
