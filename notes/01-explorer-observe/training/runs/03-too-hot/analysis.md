# Run `01-explorer-observe/training/03-too-hot`: what happened

State **finished** at step 20000 of 20000. Code commit `94eeec7`.

## Curves
* Validation loss was best at step 19500: 0.5678 (AUROC 0.010).
* At step 20000: validation loss 0.5739, training loss 0.5995, AUROC 0.010. 500 steps were run after the best point.

## Layers (latest probe)
| Layer | Gradient norm | Weight norm | Update / weight | Update peaked at step | Dead units |
|---|---|---|---|---|---|
| body.0 | 0 | 918 | 0.00e+00 | 1 | n/a |
| body.3 | 0 | 681 | 0.00e+00 | 1 | n/a |
| head | 0 | 72.6 | 0.00e+00 | 1 | n/a |

## Patients and predictions
| | First evaluation | Latest |
|---|---|---|
| Training patients on the right side of 0.5 | 0.725 | 0.725 |
| Validation patients on the right side of 0.5 | 0.744 | 0.744 |
| Training patients that were right and became wrong (count) | 0.000 | 0.000 |
| Calibration error on validation | 0.011 | 0.046 |
| Validation loss minus training loss | -0.022 | -0.026 |

## Events
* step 1 [warning] Layer body.0 changed by 100% of its own size in one step (step 1): steps this large overwrite what was learned.
* step 1 [warning] Layer body.3 changed by 100% of its own size in one step (step 1): steps this large overwrite what was learned.
* step 1 [warning] Layer head changed by 100% of its own size in one step (step 1): steps this large overwrite what was learned.
* step 25 [critical] Training loss is 1.82e+03 at step 25, 2757 times its first value (0.66): the updates are too large.
* step 250 [warning] 95% of the units of layer body.1 are zero for every validation patient at step 250: that part of the model has stopped contributing.
* step 250 [warning] 88% of the units of layer body.4 are zero for every validation patient at step 250: that part of the model has stopped contributing.
* step 350 [warning] Layer body.0 has moved by less than 1e-06 of its size per step since step 150: it is effectively frozen.
* step 350 [warning] Layer body.3 has moved by less than 1e-06 of its size per step since step 150: it is effectively frozen.
* step 350 [warning] Layer head has moved by less than 1e-06 of its size per step since step 150: it is effectively frozen.
* step 1500 [warning] Training loss is 0.5992 at step 1500, from 0.5901 at step 250: nothing is being learned. Layers move by about 0.0e+00 of their size per step.

