## Results

Chosen config (lowest pooled backtest Poisson deviance): h=1: num_leaves=127, n_estimators=600, h=3: num_leaves=127, n_estimators=600

### Backtest, 1 hour ahead (pooled over 4 folds)

| Model | RMSE | Poisson dev. | MAE | Rows | Coverage | RMSE vs best baseline | Dev. vs best baseline | MAE vs best baseline |
|---|---|---|---|---|---|---|---|---|
| Predict zero (reference) | 2.786 | 15.547 | 1.046 | 7,396,256 | 0.995 |  |  |  |
| Baseline: same hour last week | 1.931 | 3.303 | 0.843 | 7,396,256 | 0.995 |  |  |  |
| Baseline: mean of same hour, last 4 weeks | 1.592 | 1.484 | 0.737 | 7,396,256 | 0.995 |  |  |  |
| LightGBM [common] | 1.399 | 0.875 | 0.661 | 7,396,256 | 0.995 | +12.1% | +41.0% | +10.3% |
| LightGBM [all rows] | 1.429 | 0.879 | 0.663 | 7,434,121 | 1.000 |  |  |  |
| *LightGBM + weather at t+h (ablation, upper bound) [common]* | 1.396 | 0.871 | 0.659 | 7,396,256 | 0.995 | +12.3% | +41.3% | +10.6% |
| *LightGBM + weather at t+h (ablation, upper bound) [all rows]* | 1.429 | 0.875 | 0.661 | 7,434,121 | 1.000 |  |  |  |

Per fold (common subset), h=1:

| Fold | RMSE 1w | RMSE 4w | RMSE LGBM | vs best | Dev. 1w | Dev. 4w | Dev. LGBM | vs best | MAE vs best |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 1.645 | 1.385 | 1.126 | +18.8% | 2.830 | 1.255 | 0.796 | +36.6% | +18.0% |
| f2 | 0.916 | 0.781 | 0.667 | +14.6% | 2.091 | 1.072 | 0.547 | +49.0% | +8.4% |
| f3 | 2.005 | 1.668 | 1.468 | +12.0% | 3.971 | 1.790 | 0.975 | +45.5% | +7.2% |
| f4 | 2.673 | 2.169 | 1.962 | +9.5% | 4.249 | 1.794 | 1.165 | +35.1% | +8.1% |

Slices, h=1 (common subset; % = improvement over the best baseline in the slice):

| Slice | Value | Rows | RMSE LGBM | RMSE best base | RMSE | Deviance | MAE | Loses on |
|---|---|---|---|---|---|---|---|---|
| tier | low | 2,524,421 | 0.435 | 0.483 | +9.9% | +48.6% | +0.0% |  |
| tier | mid | 2,425,791 | 1.049 | 1.199 | +12.5% | +42.6% | +8.7% |  |
| tier | top | 2,446,044 | 2.152 | 2.450 | +12.2% | +36.4% | +12.5% |  |
| hour_bucket | evening peak 16-19 | 1,233,623 | 2.084 | 2.416 | +13.7% | +41.1% | +13.8% |  |
| hour_bucket | late 20-23 | 1,233,660 | 1.432 | 1.647 | +13.1% | +42.4% | +10.3% |  |
| hour_bucket | midday 10-15 | 1,850,310 | 1.532 | 1.717 | +10.8% | +40.3% | +10.4% |  |
| hour_bucket | morning peak 06-09 | 1,233,488 | 1.197 | 1.325 | +9.7% | +37.3% | +7.2% |  |
| hour_bucket | night 00-05 | 1,845,175 | 0.509 | 0.565 | +9.9% | +44.8% | +1.1% |  |
| rain | no rain | 6,974,533 | 1.418 | 1.597 | +11.2% | +41.0% | +9.3% |  |
| rain | rain | 396,419 | 1.000 | 1.500 | +33.3% | +42.0% | +30.3% |  |
| day_type | holiday | 181,872 | 1.516 | 2.186 | +30.7% | +56.3% | +34.7% |  |
| day_type | weekday | 5,110,153 | 1.373 | 1.549 | +11.4% | +39.7% | +9.3% |  |
| day_type | weekend | 2,104,231 | 1.451 | 1.635 | +11.3% | +42.1% | +10.1% |  |
| history | established | 7,355,905 | 1.370 | 1.573 | +12.9% | +41.1% | +10.4% |  |
| history | new (<8 weeks history) | 40,351 | 4.075 | 3.745 | -8.8% | +33.5% | -1.8% | ✗ rmse/mae |

**Slices where LightGBM does not beat the best baseline on RMSE or deviance:** history = new (<8 weeks history) (rmse/mae)

### Backtest, 3 hours ahead (pooled over 4 folds)

| Model | RMSE | Poisson dev. | MAE | Rows | Coverage | RMSE vs best baseline | Dev. vs best baseline | MAE vs best baseline |
|---|---|---|---|---|---|---|---|---|
| Predict zero (reference) | 2.786 | 15.548 | 1.046 | 7,396,256 | 0.995 |  |  |  |
| Baseline: same hour last week | 1.931 | 3.303 | 0.843 | 7,396,256 | 0.995 |  |  |  |
| Baseline: mean of same hour, last 4 weeks | 1.592 | 1.484 | 0.737 | 7,396,256 | 0.995 |  |  |  |
| LightGBM [common] | 1.464 | 0.904 | 0.675 | 7,396,256 | 0.995 | +8.0% | +39.1% | +8.4% |
| LightGBM [all rows] | 1.499 | 0.909 | 0.677 | 7,434,017 | 1.000 |  |  |  |
| *LightGBM + weather at t+h (ablation, upper bound) [common]* | 1.441 | 0.892 | 0.669 | 7,396,256 | 0.995 | +9.5% | +39.9% | +9.2% |
| *LightGBM + weather at t+h (ablation, upper bound) [all rows]* | 1.476 | 0.897 | 0.671 | 7,434,017 | 1.000 |  |  |  |

Per fold (common subset), h=3:

| Fold | RMSE 1w | RMSE 4w | RMSE LGBM | vs best | Dev. 1w | Dev. 4w | Dev. LGBM | vs best | MAE vs best |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 1.645 | 1.385 | 1.150 | +17.0% | 2.830 | 1.255 | 0.817 | +34.9% | +16.7% |
| f2 | 0.916 | 0.781 | 0.680 | +12.9% | 2.092 | 1.072 | 0.560 | +47.8% | +6.9% |
| f3 | 2.005 | 1.668 | 1.537 | +7.9% | 3.971 | 1.790 | 1.010 | +43.6% | +5.2% |
| f4 | 2.673 | 2.169 | 2.075 | +4.3% | 4.249 | 1.794 | 1.210 | +32.6% | +5.9% |

Slices, h=3 (common subset; % = improvement over the best baseline in the slice):

| Slice | Value | Rows | RMSE LGBM | RMSE best base | RMSE | Deviance | MAE | Loses on |
|---|---|---|---|---|---|---|---|---|
| tier | low | 2,524,421 | 0.437 | 0.483 | +9.6% | +48.2% | -0.7% | ✗ mae |
| tier | mid | 2,425,791 | 1.070 | 1.199 | +10.7% | +41.3% | +7.6% |  |
| tier | top | 2,446,044 | 2.270 | 2.450 | +7.4% | +33.3% | +10.1% |  |
| hour_bucket | evening peak 16-19 | 1,233,623 | 2.166 | 2.416 | +10.3% | +38.8% | +12.0% |  |
| hour_bucket | late 20-23 | 1,233,660 | 1.486 | 1.647 | +9.8% | +40.6% | +8.2% |  |
| hour_bucket | midday 10-15 | 1,850,310 | 1.641 | 1.717 | +4.4% | +38.2% | +8.6% |  |
| hour_bucket | morning peak 06-09 | 1,233,488 | 1.231 | 1.325 | +7.1% | +35.2% | +5.3% |  |
| hour_bucket | night 00-05 | 1,845,175 | 0.517 | 0.565 | +8.5% | +43.7% | -0.5% | ✗ mae |
| rain | no rain | 6,974,533 | 1.481 | 1.597 | +7.3% | +39.3% | +7.8% |  |
| rain | rain | 396,419 | 1.130 | 1.500 | +24.6% | +34.8% | +20.5% |  |
| day_type | holiday | 181,872 | 1.648 | 2.186 | +24.6% | +53.9% | +32.3% |  |
| day_type | weekday | 5,110,153 | 1.424 | 1.549 | +8.0% | +37.8% | +7.4% |  |
| day_type | weekend | 2,104,231 | 1.541 | 1.635 | +5.8% | +40.3% | +8.3% |  |
| history | established | 7,355,777 | 1.428 | 1.572 | +9.2% | +39.2% | +8.6% |  |
| history | new (<8 weeks history) | 40,479 | 4.638 | 3.749 | -23.7% | +19.9% | -12.4% | ✗ rmse/mae |

**Slices where LightGBM does not beat the best baseline on RMSE or deviance:** history = new (<8 weeks history) (rmse/mae)

### FINAL test, 2026-01..09, 1 hour ahead (established stations; evaluated once)

| Model | RMSE | Poisson dev. | MAE | Rows | Coverage | RMSE vs best baseline | Dev. vs best baseline | MAE vs best baseline |
|---|---|---|---|---|---|---|---|---|
| Predict zero (reference) | 3.244 | 17.623 | 1.163 | 5,604,262 | 0.999 |  |  |  |
| Baseline: same hour last week | 2.314 | 3.657 | 0.921 | 5,604,262 | 0.999 |  |  |  |
| Baseline: mean of same hour, last 4 weeks | 1.876 | 1.624 | 0.780 | 5,604,262 | 0.999 |  |  |  |
| LightGBM [common] | 1.590 | 0.910 | 0.706 | 5,604,262 | 0.999 | +15.3% | +44.0% | +9.5% |
| LightGBM [all rows] | 1.590 | 0.910 | 0.706 | 5,607,703 | 1.000 |  |  |  |

New in 2026 stations, h=1 (reported separately):

| Model | RMSE | Poisson dev. | MAE | Rows | Coverage | RMSE vs best baseline | Dev. vs best baseline | MAE vs best baseline |
|---|---|---|---|---|---|---|---|---|
| Predict zero (reference) | 2.735 | 12.620 | 0.834 | 205,176 | 0.794 |  |  |  |
| Baseline: same hour last week | 2.248 | 3.122 | 0.726 | 205,176 | 0.794 |  |  |  |
| Baseline: mean of same hour, last 4 weeks | 1.781 | 1.417 | 0.620 | 205,176 | 0.794 |  |  |  |
| LightGBM [common] | 1.478 | 0.821 | 0.579 | 205,176 | 0.794 | +17.0% | +42.0% | +6.6% |
| LightGBM [all rows] | 1.438 | 0.801 | 0.552 | 258,517 | 1.000 |  |  |  |

Final slices, h=1:

| Slice | Value | Rows | RMSE LGBM | RMSE best base | RMSE | Deviance | MAE | Loses on |
|---|---|---|---|---|---|---|---|---|
| tier | low | 1,897,676 | 0.468 | 0.517 | +9.5% | +49.8% | -1.6% | ✗ mae |
| tier | mid | 1,844,956 | 1.161 | 1.316 | +11.8% | +44.3% | +7.1% |  |
| tier | top | 1,861,630 | 2.460 | 2.934 | +16.2% | +41.4% | +12.1% |  |
| hour_bucket | evening peak 16-19 | 934,861 | 2.370 | 2.841 | +16.6% | +43.9% | +12.2% |  |
| hour_bucket | late 20-23 | 934,164 | 1.756 | 2.005 | +12.4% | +44.6% | +9.6% |  |
| hour_bucket | midday 10-15 | 1,402,593 | 1.710 | 2.087 | +18.1% | +44.9% | +11.1% |  |
| hour_bucket | morning peak 06-09 | 935,118 | 1.261 | 1.374 | +8.2% | +39.3% | +4.9% |  |
| hour_bucket | night 00-05 | 1,397,526 | 0.569 | 0.632 | +10.0% | +46.2% | +0.4% |  |
| rain | no rain | 5,231,964 | 1.613 | 1.874 | +13.9% | +44.2% | +8.3% |  |
| rain | rain | 352,014 | 1.219 | 1.940 | +37.2% | +39.6% | +26.3% |  |
| day_type | holiday | 123,538 | 2.027 | 3.099 | +34.6% | +65.1% | +27.9% |  |
| day_type | weekday | 3,882,601 | 1.540 | 1.747 | +11.8% | +41.5% | +7.6% |  |
| day_type | weekend | 1,598,123 | 1.668 | 2.046 | +18.5% | +46.2% | +11.8% |  |
| history | established | 5,604,262 | 1.590 | 1.876 | +15.3% | +44.0% | +9.5% |  |

**Final slices where LightGBM does not beat the best baseline on RMSE or deviance:** none.

### FINAL test, 2026-01..09, 3 hours ahead (established stations; evaluated once)

| Model | RMSE | Poisson dev. | MAE | Rows | Coverage | RMSE vs best baseline | Dev. vs best baseline | MAE vs best baseline |
|---|---|---|---|---|---|---|---|---|
| Predict zero (reference) | 3.244 | 17.623 | 1.163 | 5,604,262 | 0.999 |  |  |  |
| Baseline: same hour last week | 2.314 | 3.658 | 0.921 | 5,604,262 | 0.999 |  |  |  |
| Baseline: mean of same hour, last 4 weeks | 1.876 | 1.624 | 0.780 | 5,604,262 | 0.999 |  |  |  |
| LightGBM [common] | 1.688 | 0.952 | 0.728 | 5,604,262 | 0.999 | +10.0% | +41.4% | +6.7% |
| LightGBM [all rows] | 1.688 | 0.952 | 0.727 | 5,607,703 | 1.000 |  |  |  |

New in 2026 stations, h=3 (reported separately):

| Model | RMSE | Poisson dev. | MAE | Rows | Coverage | RMSE vs best baseline | Dev. vs best baseline | MAE vs best baseline |
|---|---|---|---|---|---|---|---|---|
| Predict zero (reference) | 2.735 | 12.620 | 0.834 | 205,176 | 0.794 |  |  |  |
| Baseline: same hour last week | 2.248 | 3.122 | 0.726 | 205,176 | 0.794 |  |  |  |
| Baseline: mean of same hour, last 4 weeks | 1.781 | 1.417 | 0.620 | 205,176 | 0.794 |  |  |  |
| LightGBM [common] | 1.637 | 0.885 | 0.608 | 205,176 | 0.794 | +8.1% | +37.5% | +2.0% |
| LightGBM [all rows] | 1.591 | 0.865 | 0.580 | 258,347 | 1.000 |  |  |  |

Final slices, h=3:

| Slice | Value | Rows | RMSE LGBM | RMSE best base | RMSE | Deviance | MAE | Loses on |
|---|---|---|---|---|---|---|---|---|
| tier | low | 1,897,676 | 0.472 | 0.517 | +8.7% | +49.1% | -2.6% | ✗ mae |
| tier | mid | 1,844,956 | 1.188 | 1.316 | +9.7% | +42.8% | +5.4% |  |
| tier | top | 1,861,630 | 2.637 | 2.935 | +10.1% | +37.3% | +8.7% |  |
| hour_bucket | evening peak 16-19 | 934,861 | 2.498 | 2.841 | +12.1% | +40.9% | +9.3% |  |
| hour_bucket | late 20-23 | 934,164 | 1.828 | 2.005 | +8.8% | +42.3% | +7.2% |  |
| hour_bucket | midday 10-15 | 1,402,593 | 1.883 | 2.087 | +9.7% | +41.5% | +7.8% |  |
| hour_bucket | morning peak 06-09 | 935,118 | 1.294 | 1.374 | +5.8% | +37.5% | +2.6% |  |
| hour_bucket | night 00-05 | 1,397,526 | 0.582 | 0.632 | +8.0% | +45.1% | -1.2% | ✗ mae |
| rain | no rain | 5,231,964 | 1.700 | 1.874 | +9.3% | +42.1% | +6.4% |  |
| rain | rain | 352,014 | 1.529 | 1.940 | +21.2% | +28.2% | +11.5% |  |
| day_type | holiday | 123,538 | 2.246 | 3.099 | +27.5% | +62.3% | +24.7% |  |
| day_type | weekday | 3,882,601 | 1.611 | 1.747 | +7.8% | +39.3% | +5.3% |  |
| day_type | weekend | 1,598,123 | 1.816 | 2.046 | +11.2% | +42.9% | +8.1% |  |
| history | established | 5,604,262 | 1.688 | 1.876 | +10.0% | +41.4% | +6.7% |  |

**Final slices where LightGBM does not beat the best baseline on RMSE or deviance:** none.
