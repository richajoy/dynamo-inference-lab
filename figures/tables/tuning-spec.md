| config | c | TPOT ms | out tok/s per GPU | tok/s per user | TTFT p50 ms | avg in flight | ok/err |
|---|---|---|---|---|---|---|---|
| no speculation | 1 | 6.85 | 145 | 146.0 | 43 | 1 | 32/0 |
| no speculation | 4 | 7.01 | 567 | 142.6 | 43 | 4 | 32/0 |
| no speculation | 16 | 7.59 | 2,083 | 131.8 | 97 | 16 | 64/0 |
| no speculation | 64 | 9.94 | 6,278 | 100.6 | 249 | 64 | 256/0 |
| speculative, n-gram k=4 | 1 | 3.22 | 307 | 347.1 | 43 | 1 | 32/0 |
| speculative, n-gram k=4 | 4 | 3.51 | 1,053 | 322.5 | 42 | 4 | 32/0 |
| speculative, n-gram k=4 | 16 | 4.21 | 3,166 | 268.0 | 43 | 14 (!) | 64/0 |
| speculative, n-gram k=4 | 64 | 5.72 | 9,077 | 196.0 | 42 | 53 (!) | 256/0 |
| speculative, Qwen3-0.6B draft k=4 | 1 | 4.38 | 226 | 239.6 | 43 | 1 | 32/0 |
| speculative, Qwen3-0.6B draft k=4 | 4 | 4.53 | 845 | 229.9 | 52 | 4 | 32/0 |
| speculative, Qwen3-0.6B draft k=4 | 16 | 4.88 | 2,902 | 213.2 | 54 | 14 | 64/0 |
| speculative, Qwen3-0.6B draft k=4 | 64 | 7.68 | 6,850 | 135.1 | 94 | 54 (!) | 256/0 |
