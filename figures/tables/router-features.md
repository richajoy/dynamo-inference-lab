| arm | extra flags | hit rate | prefix affinity | TTFT p50 ms | TTFT p90 ms | ITL ms | router time ms | max active prefill tokens/worker |
|---|---|---|---|---|---|---|---|---|
| kv defaults | kv | 0.764 | 0.70 | 81.3 | 403.9 | 9.32 | 0.07 | 11,146 |
| + load scale 10 | --router-prefill-load-scale 10 | 0.860 | 0.89 | 58.7 | 352.1 | 8.78 | 0.10 | 2,567 |
| + load scale 10 + queue threshold 0.5 | --router-prefill-load-scale 10 --router-queue-threshold 0.5 | 0.831 | 0.84 | 48.9 | 143.7 | 9.49 | 33.97 | 4,291 |
| + load scale 10 + credit decay 1.0 | --router-prefill-load-scale 10 --router-kv-overlap-score-credit-decay 1.0 | 0.715 | 0.69 | 88.8 | 287.1 | 9.90 | 0.08 | 194 |
| + load scale 10 + temperature 0.5 | --router-prefill-load-scale 10 --router-temperature 0.5 | 0.691 | 0.65 | 73.1 | 278.8 | 10.43 | 0.09 | 4,291 |
