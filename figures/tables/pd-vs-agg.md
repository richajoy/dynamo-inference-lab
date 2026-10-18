| shape | c | deployment (2 GPUs each) | req/s | TTFT p50 ms | TTFT p99 ms | TPOT ms | out tok/s per GPU | tok/s per user | avg in flight | ok/err |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill-heavy | 16 | aggregated x2 + KV router | 9.93 | 449 | 594 | 9.56 | 635 | 106.0 | 16 | 64/0 |
| prefill-heavy | 16 | P/D 1P1D, KV over TCP | 0.92 | 16,135 | 16,291 | 9.39 | 59 | 106.5 | 16 | 64/0 |
| prefill-heavy | 16 | P/D 1P1D, KV over NVLink | 9.98 | 205 | 1,234 | 8.82 | 638 | 113.6 | 15 | 64/0 |
| prefill-heavy | 16 | P/D 1P1D, NVLink, KV block 64 | 9.98 | 205 | 1,226 | 8.84 | 639 | 113.2 | 15 | 64/0 |
| prefill-heavy | 64 | aggregated x2 + KV router | 16.82 | 603 | 2,394 | 23.79 | 1,076 | 43.6 | 64 | 256/0 |
| prefill-heavy | 64 | P/D 1P1D, KV over TCP | 0.92 | 65,078 | 118,412 | 11.71 | 59 | 91.9 | 61 | 256/0 |
| prefill-heavy | 64 | P/D 1P1D, KV over NVLink | 12.84 | 2,878 | 4,697 | 12.70 | 822 | 81.1 | 58 | 256/0 |
| prefill-heavy | 64 | P/D 1P1D, NVLink, KV block 64 | 12.87 | 3,315 | 4,587 | 10.21 | 824 | 98.2 | 57 (!) | 256/0 |
| decode-heavy | 16 | aggregated x2 + KV router | 2.10 | 44 | 83 | 7.31 | 1,078 | 136.8 | 16 | 64/0 |
| decode-heavy | 16 | P/D 1P1D, KV over TCP | 1.71 | 1,453 | 1,653 | 7.67 | 875 | 130.4 | 16 | 64/0 |
| decode-heavy | 16 | P/D 1P1D, KV over NVLink | 2.07 | 151 | 178 | 7.39 | 1,061 | 135.3 | 16 | 64/0 |
| decode-heavy | 64 | aggregated x2 + KV router | 7.30 | 138 | 209 | 8.36 | 3,736 | 119.6 | 63 | 256/0 |
| decode-heavy | 64 | P/D 1P1D, KV over TCP | 4.14 | 5,422 | 6,345 | 9.58 | 2,120 | 104.7 | 61 | 256/0 |
| decode-heavy | 64 | P/D 1P1D, KV over NVLink | 6.19 | 306 | 587 | 9.71 | 3,166 | 103.0 | 64 | 256/0 |
| long | 16 | aggregated x2 + KV router | 1.46 | 741 | 1,277 | 10.04 | 747 | 99.7 | 16 | 64/0 |
| long | 16 | P/D 1P1D, KV over TCP | 0.33 | 35,466 | 35,641 | 12.12 | 171 | 82.5 | 16 | 64/0 |
| long | 16 | P/D 1P1D, KV over NVLink | 1.19 | 310 | 2,540 | 12.30 | 610 | 81.3 | 16 | 64/0 |
| long | 16 | P/D 1P1D, NVLink, KV block 64 | 1.24 | 246 | 2,552 | 11.79 | 635 | 84.9 | 16 | 64/0 |
| long | 64 | aggregated x2 + KV router | 2.85 | 867 | 5,053 | 20.64 | 1,460 | 48.6 | 64 | 256/0 |
| long | 64 | P/D 1P1D, KV over TCP | 0.38 | 146,379 | 179,243 | 15.46 | 39 | 74.6 | 63 | 68/188 |
| long | 64 | P/D 1P1D, KV over NVLink | 1.87 | 1,637 | 18,913 | 28.07 | 957 | 36.6 | 60 | 256/0 |
| long | 64 | P/D 1P1D, NVLink, KV block 64 | 1.51 | 5,274 | 42,244 | 21.22 | 236 | 47.3 | 64 | 64/192 |
