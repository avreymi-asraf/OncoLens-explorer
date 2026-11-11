# Run `01-explorer-observe/training/02-overfit`: what happened

State **finished** at step 60000 of 60000. Code commit `94eeec7`.

## Curves
* Validation loss was best at step 500: 0.4612 (AUROC 0.794).
* At step 60000: validation loss 2.0422, training loss 0.0142, AUROC 0.794. 59500 steps were run after the best point.

## Layers (latest probe)
| Layer | Gradient norm | Weight norm | Update / weight | Update peaked at step | Dead units |
|---|---|---|---|---|---|
| body.0 | 0.402 | 21 | 3.46e-04 | 1 | n/a |
| body.3 | 0.143 | 37.3 | 2.03e-04 | 1 | n/a |
| head | 0.0119 | 12.3 | 6.18e-05 | 1 | n/a |

## Patients and predictions
| | First evaluation | Latest |
|---|---|---|
| Training patients on the right side of 0.5 | 0.797 | 0.993 |
| Validation patients on the right side of 0.5 | 0.767 | 0.770 |
| Training patients that were right and became wrong (count) | 0.000 | 2200.000 |
| Calibration error on validation | 0.071 | 0.204 |
| Validation loss minus training loss | 0.033 | 2.028 |

## Events
* step 2000 [warning] Validation loss was best at step 500 (0.4612) and is 0.5327 at step 2000, while training loss went from 0.4117 to 0.2610: the model is fitting the training patients only.

