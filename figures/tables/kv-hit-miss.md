| GPU | arm | prefix affinity | hit n | hit TTFT p50 ms | miss n | miss TTFT p50 ms | miss TTFT p90 ms |
|---|---|---|---|---|---|---|---|
| A100 | kv, no events (TTL) | 0.64 | 720 | 109.9 | 84 | 416.5 | 956.1 |
| A100 | kv + load scale 4 | 0.76 | 657 | 87.3 | 147 | 264.5 | 554.5 |
| A100 | kv + load scale 10 | 0.96 | 744 | 86.7 | 60 | 381.4 | 1,320 |
| A100 | no events (TTL) + load scale 10 | 0.96 | 755 | 87.8 | 49 | 411.4 | 1,311 |
| A100 | no events (LRU) + load scale 10 | 0.93 | 737 | 86.1 | 67 | 383.3 | 1,132 |
| H100 | round-robin | 0.57 | 451 | 22.8 | 353 | 43.9 | 63.0 |
| H100 | kv (defaults) | 0.63 | 530 | 30.2 | 274 | 44.9 | 110.3 |
| H100 | kv + load scale 10 | 0.93 | 741 | 30.6 | 63 | 49.9 | 187.1 |
