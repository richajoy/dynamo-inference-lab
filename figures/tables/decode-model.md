| point | batch | avg context | predicted ms | measured ms | error |
|---|---|---|---|---|---|
| decode-heavy c1 | 1 | 768 | 6.85 | 6.85 | 0% |
| decode-heavy c4 | 4 | 768 | 7.00 | 7.01 | 0% |
| decode-heavy c16 | 16 | 768 | 7.61 | 7.59 | 0% |
| decode-heavy c64 | 64 | 768 | 10.05 | 9.94 | -1% |
| prefill-heavy c64 (median gap) | ~53 | ~3,064 | 17.53 | 16.96 | -3% |
