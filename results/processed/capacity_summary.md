# Capacity sweep summary

> Cells marked as LOW_N have fewer than 10 runs and are excluded from mean/SD reporting; no value is fabricated or interpolated.

## Arm A (full cohort)

| hidden_units | configuration | n_runs | MAE mean | MAE SD | MAE min | MAE max | mean R² | flag |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4 | FULL_HYBRID | 10 | 0.3764 | 0.0215 | 0.3406 | 0.4010 | 0.6086 |  |
| 4 | NO_STATIC_MECH | 10 | 0.3605 | 0.0172 | 0.3381 | 0.3971 | 0.6134 |  |
| 4 | NO_TEMPORAL | 10 | 0.3610 | 0.0233 | 0.3302 | 0.4130 | 0.6092 |  |
| 4 | ONLY_MECH | 10 | 0.3583 | 0.0212 | 0.3428 | 0.4146 | 0.6059 |  |
| 4 | PURE_ML | 10 | 0.3739 | 0.0168 | 0.3481 | 0.4074 | 0.6027 |  |
| 8 | FULL_HYBRID | 10 | 0.3677 | 0.0200 | 0.3435 | 0.3999 | 0.6130 |  |
| 8 | NO_STATIC_MECH | 10 | 0.3599 | 0.0198 | 0.3398 | 0.3905 | 0.6135 |  |
| 8 | NO_TEMPORAL | 10 | 0.3709 | 0.0162 | 0.3473 | 0.3943 | 0.6076 |  |
| 8 | ONLY_MECH | 10 | 0.3634 | 0.0269 | 0.3312 | 0.4102 | 0.6105 |  |
| 8 | PURE_ML | 10 | 0.3763 | 0.0122 | 0.3624 | 0.3975 | 0.6034 |  |
| 16 | FULL_HYBRID | 10 | 0.3719 | 0.0216 | 0.3350 | 0.4115 | 0.6120 |  |
| 16 | NO_STATIC_MECH | 10 | 0.3824 | 0.0270 | 0.3386 | 0.4222 | 0.6058 |  |
| 16 | NO_TEMPORAL | 10 | 0.3820 | 0.0366 | 0.3307 | 0.4513 | 0.6029 |  |
| 16 | ONLY_MECH | 10 | 0.3836 | 0.0275 | 0.3290 | 0.4241 | 0.6001 |  |
| 16 | PURE_ML | 10 | 0.3920 | 0.0303 | 0.3543 | 0.4453 | 0.5950 |  |
| 32 | FULL_HYBRID | 10 | 0.4031 | 0.0496 | 0.3632 | 0.5327 | 0.5875 |  |
| 32 | NO_STATIC_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 32 | NO_TEMPORAL | 6 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 32 | ONLY_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 32 | PURE_ML | 10 | 0.4058 | 0.0309 | 0.3627 | 0.4593 | 0.5837 |  |

### FULL_HYBRID vs PURE_ML comparison for arm A (full cohort)

| hidden_units | FULL_HYBRID MAE mean | PURE_ML MAE mean | absolute diff | percent improvement | paired Wilcoxon p | Holm p | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.3764 | 0.3739 | 0.0025 | 0.67% | 0.695 | 1.000 | matched runs |
| 8 | 0.3677 | 0.3763 | 0.0086 | -2.33% | 0.275 | 1.000 | matched runs |
| 16 | 0.3719 | 0.3920 | 0.0201 | -5.40% | 0.049 | 0.537 | matched runs |
| 32 | 0.4031 | 0.4058 | 0.0027 | -0.67% | 0.846 | 1.000 | matched runs |

## Arm B (interior cohort)

| hidden_units | configuration | n_runs | MAE mean | MAE SD | MAE min | MAE max | mean R² | flag |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4 | FULL_HYBRID | 10 | 0.3471 | 0.0152 | 0.3345 | 0.3834 | 0.4285 |  |
| 4 | NO_STATIC_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 4 | NO_TEMPORAL | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 4 | ONLY_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 4 | PURE_ML | 10 | 0.3698 | 0.0230 | 0.3481 | 0.4129 | 0.3995 |  |
| 8 | FULL_HYBRID | 10 | 0.3639 | 0.0272 | 0.3379 | 0.4262 | 0.4164 |  |
| 8 | NO_STATIC_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 8 | NO_TEMPORAL | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 8 | ONLY_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 8 | PURE_ML | 10 | 0.3768 | 0.0145 | 0.3585 | 0.3950 | 0.3988 |  |
| 16 | FULL_HYBRID | 10 | 0.3705 | 0.0214 | 0.3390 | 0.3985 | 0.4119 |  |
| 16 | NO_STATIC_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 16 | NO_TEMPORAL | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 16 | ONLY_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 16 | PURE_ML | 10 | 0.3858 | 0.0266 | 0.3554 | 0.4232 | 0.3844 |  |
| 32 | FULL_HYBRID | 10 | 0.3970 | 0.0226 | 0.3557 | 0.4292 | 0.3708 |  |
| 32 | NO_STATIC_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 32 | NO_TEMPORAL | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 32 | ONLY_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 32 | PURE_ML | 10 | 0.3945 | 0.0157 | 0.3698 | 0.4202 | 0.3717 |  |

### FULL_HYBRID vs PURE_ML comparison for arm B (interior cohort)

| hidden_units | FULL_HYBRID MAE mean | PURE_ML MAE mean | absolute diff | percent improvement | paired Wilcoxon p | Holm p | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.3471 | 0.3698 | 0.0227 | -6.53% | 0.002 | 0.023 | matched runs |
| 8 | 0.3639 | 0.3768 | 0.0128 | -3.52% | 0.160 | 1.000 | matched runs |
| 16 | 0.3705 | 0.3858 | 0.0154 | -4.15% | 0.131 | 1.000 | matched runs |
| 32 | 0.3970 | 0.3945 | 0.0025 | 0.63% | 0.695 | 1.000 | matched runs |

## Arm C (matched cohort)

| hidden_units | configuration | n_runs | MAE mean | MAE SD | MAE min | MAE max | mean R² | flag |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4 | FULL_HYBRID | 10 | 0.3704 | 0.0070 | 0.3583 | 0.3772 | 0.4814 |  |
| 4 | NO_STATIC_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 4 | NO_TEMPORAL | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 4 | ONLY_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 4 | PURE_ML | 10 | 0.3781 | 0.0140 | 0.3656 | 0.4130 | 0.4685 |  |
| 8 | FULL_HYBRID | 10 | 0.3863 | 0.0173 | 0.3543 | 0.4122 | 0.4719 |  |
| 8 | NO_STATIC_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 8 | NO_TEMPORAL | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 8 | ONLY_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 8 | PURE_ML | 10 | 0.3913 | 0.0228 | 0.3652 | 0.4328 | 0.4612 |  |
| 16 | FULL_HYBRID | 10 | 0.4029 | 0.0345 | 0.3544 | 0.4558 | 0.4470 |  |
| 16 | NO_STATIC_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 16 | NO_TEMPORAL | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 16 | ONLY_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 16 | PURE_ML | 10 | 0.4296 | 0.0342 | 0.3744 | 0.4926 | 0.4144 |  |
| 32 | FULL_HYBRID | 10 | 0.4399 | 0.0429 | 0.4017 | 0.5519 | 0.3953 |  |
| 32 | NO_STATIC_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 32 | NO_TEMPORAL | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 32 | ONLY_MECH | 0 | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N | LOW_N |
| 32 | PURE_ML | 10 | 0.4197 | 0.0288 | 0.3720 | 0.4851 | 0.4245 |  |

### FULL_HYBRID vs PURE_ML comparison for arm C (matched cohort)

| hidden_units | FULL_HYBRID MAE mean | PURE_ML MAE mean | absolute diff | percent improvement | paired Wilcoxon p | Holm p | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.3704 | 0.3781 | 0.0077 | -2.07% | 0.275 | 1.000 | matched runs |
| 8 | 0.3863 | 0.3913 | 0.0050 | -1.29% | 0.846 | 1.000 | matched runs |
| 16 | 0.4029 | 0.4296 | 0.0267 | -6.62% | 0.375 | 1.000 | matched runs |
| 32 | 0.4399 | 0.4197 | 0.0203 | 4.60% | 0.275 | 1.000 | matched runs |

