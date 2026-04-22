# C1 Calibration Report

## Setup

- Dataset: `D:\Codes\face\ai_deepfake\dataset_final\test`
- Samples (`status=ok`): `450`
- Signal under analysis: `fake_probability`

## Summary Metrics

- Brier score: `0.055036`
- ECE (10 bins): `0.149655`

## Reliability Table (10 bins)

| bin | count | avg_prob | empirical_fake_rate | abs_gap |
| --- | --- | --- | --- | --- |
| [0.0,0.1) | 40 | 0.074509 | 0.000000 | 0.074509 |
| [0.1,0.2) | 92 | 0.149793 | 0.000000 | 0.149793 |
| [0.2,0.3) | 43 | 0.241947 | 0.000000 | 0.241947 |
| [0.3,0.4) | 22 | 0.335424 | 0.045455 | 0.289969 |
| [0.4,0.5) | 13 | 0.454548 | 0.384615 | 0.069932 |
| [0.5,0.6) | 16 | 0.539275 | 0.187500 | 0.351775 |
| [0.6,0.7) | 6 | 0.646928 | 0.500000 | 0.146928 |
| [0.7,0.8) | 39 | 0.755168 | 0.871795 | 0.116627 |
| [0.8,0.9) | 125 | 0.856979 | 1.000000 | 0.143021 |
| [0.9,1.0] | 54 | 0.926755 | 1.000000 | 0.073245 |

## Interpretation

- Calibration appears weak: avoid interpreting fake_probability as calibrated confidence without further calibration.
- This report is purely measured from current detector scores; no calibration model was fitted in C1.
