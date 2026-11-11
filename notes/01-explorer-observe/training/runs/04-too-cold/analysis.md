# Run `01-explorer-observe/training/04-too-cold`: what happened

State **finished** at step 20000 of 20000. Code commit `7913ab6`.

## Curves
* Validation loss was best at step 20000: 0.6577 (AUROC 0.338).
* At step 20000: validation loss 0.6577, training loss 0.6601, AUROC 0.338. 0 steps were run after the best point.

## Layers (latest probe)
| Layer | Gradient norm | Weight norm | Update / weight | Update peaked at step | Dead units |
|---|---|---|---|---|---|
| body.0 | 0.0497 | 4.59 | 7.34e-08 | 1 | 0% |
| body.3 | 0.0847 | 3.25 | 9.08e-08 | 1 | 9% |
| head | 0.0774 | 0.515 | 1.04e-07 | 14500 | n/a |

## Patients and predictions
| | First evaluation | Latest |
|---|---|---|
| Training patients on the right side of 0.5 | 0.725 | 0.725 |
| Validation patients on the right side of 0.5 | 0.742 | 0.742 |
| Training patients that were right and became wrong (count) | 0.000 | 0.000 |
| Calibration error on validation | 0.200 | 0.199 |
| Validation loss minus training loss | -0.002 | -0.002 |

## Events
* step 200 [warning] Layer body.0 has moved by less than 1e-06 of its size per step since step 1: it is effectively frozen.
* step 200 [warning] Layer body.3 has moved by less than 1e-06 of its size per step since step 1: it is effectively frozen.
* step 200 [warning] Layer head has moved by less than 1e-06 of its size per step since step 1: it is effectively frozen.
* step 1500 [warning] Training loss is 0.6606 at step 1500, from 0.6607 at step 250: nothing is being learned. Layers move by about 8.6e-08 of their size per step.

