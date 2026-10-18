| shape | c | config | req/s | TTFT p50 ms | TTFT p99 ms | TPOT ms | out tok/s per GPU | tok/s per user | avg in flight | ok/err |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill-heavy | 16 | chunk budget 512 | 7.49 | 164 | 1,442 | 13.60 | 958 | 74.6 | 15 | 64/0 |
| prefill-heavy | 16 | BF16 baseline (chunk 8192 = H100 default) | 6.85 | 639 | 1,179 | 13.63 | 877 | 74.9 | 16 | 64/0 |
| prefill-heavy | 16 | chunk 8192 set explicitly (repeat of baseline) | 6.80 | 648 | 1,195 | 13.68 | 871 | 74.6 | 16 | 64/0 |
| prefill-heavy | 64 | chunk budget 512 | 8.80 | 4,755 | 6,512 | 17.62 | 1,126 | 57.4 | 59 | 256/0 |
| prefill-heavy | 64 | BF16 baseline (chunk 8192 = H100 default) | 9.55 | 618 | 4,783 | 43.68 | 1,222 | 24.4 | 64 | 256/0 |
| prefill-heavy | 64 | chunk 8192 set explicitly (repeat of baseline) | 9.48 | 628 | 4,825 | 44.01 | 1,213 | 24.2 | 64 | 256/0 |
| decode-heavy | 16 | chunk budget 512 | 2.03 | 49 | 129 | 7.64 | 2,077 | 130.8 | 16 | 64/0 |
| decode-heavy | 16 | BF16 baseline (chunk 8192 = H100 default) | 2.03 | 100 | 110 | 7.59 | 2,083 | 131.8 | 16 | 64/0 |
| decode-heavy | 16 | chunk 8192 set explicitly (repeat of baseline) | 2.03 | 96 | 117 | 7.60 | 2,080 | 131.7 | 16 | 64/0 |
| decode-heavy | 64 | chunk budget 512 | 6.14 | 72 | 486 | 10.03 | 6,287 | 99.7 | 64 | 256/0 |
| decode-heavy | 64 | BF16 baseline (chunk 8192 = H100 default) | 6.14 | 245 | 391 | 9.95 | 6,281 | 100.5 | 64 | 256/0 |
| decode-heavy | 64 | chunk 8192 set explicitly (repeat of baseline) | 6.12 | 256 | 406 | 9.95 | 6,261 | 100.5 | 64 | 256/0 |
| long | 16 | chunk budget 512 | 1.10 | 329 | 3,290 | 13.33 | 1,127 | 75.1 | 16 | 64/0 |
| long | 16 | BF16 baseline (chunk 8192 = H100 default) | 1.08 | 848 | 2,504 | 13.58 | 1,105 | 73.7 | 16 | 64/0 |
| long | 16 | chunk 8192 set explicitly (repeat of baseline) | 1.08 | 859 | 2,547 | 13.59 | 1,104 | 73.7 | 16 | 64/0 |
| long | 64 | chunk budget 512 | 1.66 | 3,279 | 27,952 | 30.27 | 1,700 | 33.7 | 60 | 256/0 |
| long | 64 | BF16 baseline (chunk 8192 = H100 default) | 1.57 | 2,096 | 24,125 | 33.65 | 1,608 | 31.5 | 60 | 256/0 |
| long | 64 | chunk 8192 set explicitly (repeat of baseline) | 1.57 | 2,168 | 24,126 | 33.73 | 1,603 | 31.4 | 60 | 256/0 |
