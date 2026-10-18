| shape | c | config | req/s | TTFT p50 ms | TTFT p99 ms | TPOT ms | out tok/s per GPU | tok/s per user | avg in flight | ok/err |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill-heavy | 16 | BF16 baseline (chunk 8192 = H100 default) | 6.85 | 639 | 1,179 | 13.63 | 877 | 74.9 | 16 | 64/0 |
| prefill-heavy | 16 | max-num-seqs 128 | 6.73 | 723 | 1,203 | 13.53 | 862 | 75.5 | 16 | 64/0 |
| prefill-heavy | 16 | combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs) | 9.98 | 125 | 946 | 10.61 | 1,278 | 96.2 | 16 | 64/0 |
| prefill-heavy | 64 | BF16 baseline (chunk 8192 = H100 default) | 9.55 | 618 | 4,783 | 43.68 | 1,222 | 24.4 | 64 | 256/0 |
| prefill-heavy | 64 | max-num-seqs 128 | 9.44 | 625 | 4,790 | 44.16 | 1,208 | 24.1 | 64 | 256/0 |
| prefill-heavy | 64 | combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs) | 12.95 | 1,661 | 4,318 | 22.93 | 1,657 | 45.8 | 61 | 256/0 |
| prefill-heavy | 128 | max-num-seqs 128 | 9.86 | 871 | 9,785 | 85.18 | 1,262 | 13.4 | 125 | 512/0 |
| prefill-heavy | 128 | combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs) | 13.14 | 6,458 | 8,919 | 23.77 | 1,682 | 43.2 | 118 | 512/0 |
| decode-heavy | 16 | BF16 baseline (chunk 8192 = H100 default) | 2.03 | 100 | 110 | 7.59 | 2,083 | 131.8 | 16 | 64/0 |
| decode-heavy | 16 | max-num-seqs 128 | 2.03 | 96 | 107 | 7.61 | 2,079 | 131.4 | 16 | 64/0 |
| decode-heavy | 16 | combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs) | 2.94 | 51 | 89 | 5.26 | 3,009 | 190.2 | 16 | 64/0 |
| decode-heavy | 64 | BF16 baseline (chunk 8192 = H100 default) | 6.14 | 245 | 391 | 9.95 | 6,281 | 100.5 | 64 | 256/0 |
| decode-heavy | 64 | max-num-seqs 128 | 6.11 | 293 | 409 | 9.97 | 6,260 | 100.3 | 64 | 256/0 |
| decode-heavy | 64 | combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs) | 8.66 | 76 | 322 | 7.11 | 8,859 | 140.6 | 64 | 256/0 |
| decode-heavy | 128 | max-num-seqs 128 | 9.20 | 436 | 782 | 13.16 | 9,425 | 76.0 | 128 | 512/0 |
| decode-heavy | 128 | combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs) | 12.49 | 126 | 689 | 9.75 | 12,792 | 102.6 | 128 | 512/0 |
| long | 16 | BF16 baseline (chunk 8192 = H100 default) | 1.08 | 848 | 2,504 | 13.58 | 1,105 | 73.7 | 16 | 64/0 |
| long | 16 | max-num-seqs 128 | 1.08 | 859 | 2,551 | 13.61 | 1,101 | 73.6 | 16 | 64/0 |
| long | 16 | combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs) | 1.63 | 243 | 2,096 | 9.10 | 1,664 | 110.2 | 16 | 64/0 |
| long | 64 | BF16 baseline (chunk 8192 = H100 default) | 1.57 | 2,096 | 24,125 | 33.65 | 1,608 | 31.5 | 60 | 256/0 |
| long | 64 | max-num-seqs 128 | 1.56 | 2,151 | 23,098 | 33.83 | 1,598 | 31.3 | 60 | 256/0 |
| long | 64 | combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs) | 2.51 | 384 | 10,536 | 22.90 | 2,565 | 44.2 | 63 | 256/0 |
| long | 128 | max-num-seqs 128 | 1.60 | 39,556 | 73,783 | 34.73 | 1,638 | 29.5 | 119 | 512/0 |
| long | 128 | combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs) | 2.61 | 896 | 27,146 | 42.71 | 2,668 | 24.2 | 124 | 512/0 |
