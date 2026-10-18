| variant | 10 s log intervals | MiB per transfer (mean) | descriptors per transfer | transfer ms | post ms | post share | GB/s (decimal) |
|---|---|---|---|---|---|---|---|
| TCP (1 GPU per pod) | 3 | 846.0 | 44.0 | 1,986 | 0.61 | 0.0% | 0.45 |
| TCP + hostIPC | 2 | 846.0 | 45.0 | 1,990 | 0.72 | 0.0% | 0.45 |
| MIG slices (TCP) | 3 | 846.0 | 72.0 | 2,001 | 1.28 | 0.1% | 0.44 |
| NVLink, idle | 1 | 846.0 | 36.0 | 5.4 | 1.07 | 19.7% | 163.58 |
| NVLink, c=64 load, block 16 | 3 | 843.8 | 13,442 | 92.7 | 51.29 | 55.3% | 9.55 |
| NVLink, c=64 load, block 64 | 13 | 715.8 | 1,854 | 20.6 | 7.87 | 38.2% | 37.67 |
