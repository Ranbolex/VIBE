# VIBE Core

This anonymous source package isolates the VIBE criterion and edge-selection
components from the experiment training framework. It includes no datasets,
checkpoints, experiment results, map binaries, machine-specific settings, or
training hyperparameters.

## Modules

- `CandidateBuilder`: bounded joint-action candidates; an optional callback
  supplies POW-style `Q_r` scores.
- `VIBETeacher`: centralized diagnostic and candidate-based edge scores.
- `PairwiseBeliefEncoder`: receiver-local predictions of teammate actions.
- `EdgeStudent`: directed edge logits from receiver-local information.
- `GraphMessage`: sparse aggregation of explicit graph payloads.
- `losses` and `metrics`: edge supervision and diagnostic measurements.

The benchmark teacher is an estimator over supplied candidate values, not an
exact joint-action oracle. The `pow_qr` candidate mode needs a scoring callback
from the surrounding training system. `GraphMessage` also retains an explicit
`oracle_sender_hidden` diagnostic mode; deployable receiver-local use must pass
`payload_mode="receiver_local_ally"` and a receiver-local pairwise payload.
This package does not include the POW/PyMARL trainer, SMAC interface, or a
reproduction of paper win-rate results.

## Install and Test

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The core API depends only on PyTorch. The code was extracted without changing
the scoring or neural-network behavior; the included tests exercise its public
interfaces in isolation.

## Information Scope

The centralized teacher may consume centralized training-time values. The
student and belief encoder consume receiver-local inputs. For a receiver-local
message, the caller supplies ally features observable by the receiver. The
caller is responsible for enforcing this information boundary when integrating
the modules with an environment or policy trainer.

## Attribution and Anonymous Review

The source originated in an MIT-licensed POW-QMIX/PyMARL2 codebase. The
required MIT permission and disclaimer text are retained in `LICENSE`; no
author or institution identity is included in this release. See
`ANONYMITY_CHECKLIST.md` before publishing this directory.
