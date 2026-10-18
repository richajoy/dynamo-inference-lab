| shape | c | config | req/s | TTFT p50 ms | TTFT p99 ms | TPOT ms | out tok/s per GPU | tok/s per user | avg in flight | ok/err |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill-heavy | 16 | BF16 baseline (chunk 8192 = H100 default) | 6.85 | 639 | 1,179 | 13.63 | 877 | 74.9 | 16 | 64/0 |
| prefill-heavy | 16 | FP8 weights | 7.48 | 208 | 1,245 | 11.73 | 958 | 109.4 | 14 (!) | 64/0 |
| prefill-heavy | 16 | FP8 weights + fusion passes | 6.77 | 537 | 904 | 14.71 | 866 | 73.2 | 16 | 64/0 |
| prefill-heavy | 16 | FP8 KV cache | 7.09 | 675 | 1,200 | 12.71 | 907 | 80.6 | 16 | 64/0 |
| prefill-heavy | 64 | BF16 baseline (chunk 8192 = H100 default) | 9.55 | 618 | 4,783 | 43.68 | 1,222 | 24.4 | 64 | 256/0 |
| prefill-heavy | 64 | FP8 weights | 9.55 | 366 | 5,354 | 26.19 | 1,222 | 58.0 | 47 (!) | 256/0 |
| prefill-heavy | 64 | FP8 weights + fusion passes | 9.25 | 899 | 5,443 | 35.70 | 1,184 | 30.8 | 63 | 256/0 |
| prefill-heavy | 64 | FP8 KV cache | 9.99 | 636 | 4,861 | 41.25 | 1,278 | 26.3 | 64 | 256/0 |
| decode-heavy | 16 | BF16 baseline (chunk 8192 = H100 default) | 2.03 | 100 | 110 | 7.59 | 2,083 | 131.8 | 16 | 64/0 |
| decode-heavy | 16 | FP8 weights | 2.69 | 43 | 67 | 5.41 | 2,752 | 186.3 | 15 | 64/0 |
| decode-heavy | 16 | FP8 weights + fusion passes | 2.66 | 87 | 95 | 5.80 | 2,720 | 172.8 | 16 | 64/0 |
| decode-heavy | 16 | FP8 KV cache | 2.07 | 102 | 116 | 7.46 | 2,118 | 134.1 | 16 | 64/0 |
| decode-heavy | 64 | BF16 baseline (chunk 8192 = H100 default) | 6.14 | 245 | 391 | 9.95 | 6,281 | 100.5 | 64 | 256/0 |
| decode-heavy | 64 | FP8 weights | 6.13 | 121 | 8,004 | 6.69 | 6,280 | 151.7 | 51 (!) | 256/0 |
| decode-heavy | 64 | FP8 weights + fusion passes | 5.30 | 3,365 | 8,103 | 7.88 | 5,426 | 127.0 | 58 | 256/0 |
| decode-heavy | 64 | FP8 KV cache | 6.57 | 273 | 412 | 9.25 | 6,724 | 108.1 | 64 | 256/0 |
| long | 16 | BF16 baseline (chunk 8192 = H100 default) | 1.08 | 848 | 2,504 | 13.58 | 1,105 | 73.7 | 16 | 64/0 |
| long | 16 | FP8 weights | 1.22 | 269 | 1,301 | 10.23 | 1,252 | 109.4 | 13 (!) | 64/0 |
| long | 16 | FP8 weights + fusion passes | 0.99 | 649 | 1,970 | 15.03 | 1,016 | 68.5 | 16 | 64/0 |
| long | 16 | FP8 KV cache | 1.24 | 865 | 2,549 | 11.61 | 1,272 | 86.3 | 16 | 64/0 |
| long | 64 | BF16 baseline (chunk 8192 = H100 default) | 1.57 | 2,096 | 24,125 | 33.65 | 1,608 | 31.5 | 60 | 256/0 |
| long | 64 | FP8 weights | 1.81 | 595 | 31,070 | 21.32 | 1,857 | 53.2 | 50 (!) | 256/0 |
| long | 64 | FP8 weights + fusion passes | 1.54 | 2,488 | 31,693 | 30.70 | 1,580 | 33.0 | 62 | 256/0 |
| long | 64 | FP8 KV cache | 2.14 | 915 | 10,238 | 27.15 | 2,195 | 37.2 | 64 | 256/0 |
