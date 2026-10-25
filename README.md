# Ant colony foraging: where does the representation live?

Code, raw results and figures for the CS 6795 project *Distributed Cognition in
Ant Colony Foraging* (Computational Model/Tool track).

A simulated colony of 500 ants forages on a grid. Each ant knows only where it
is, which way it faces, whether it carries food, and how many steps it has taken
since its last stop. Everything task-specific lives in two scents on the ground.
The experiments ask whether such a colony can pick the shorter of two routes,
abandon it when it is sealed, and what each part of the model contributes.

## Setup

Python 3.10+ (developed on 3.12).

    pip install -r requirements.txt

## Run

    python viewer.py                 # live animation (shelf world)
    python viewer.py --seal 5000     # seal the short gap at tick 5000
    python viewer.py --control       # equal-length routes

    python final_experiments.py      # all 150 trials; about 25 min on one CPU.
                                     # Resumable: finished trials in results.jsonl are skipped.
    python analyze2.py               # statistics quoted in the report
    python make_figures.py           # the four report figures (writes ./figures)

`results/` holds the raw output of the 150 trials used in the report
(`results.jsonl`, plus the Figure 1 snapshots), so `analyze2.py` and
`make_figures.py` run as-is without rerunning the experiments.

## Files

| File | Purpose |
|---|---|
| `ants.py` | Agent model, two scent fields, world builders |
| `trial.py` | One experimental trial: world, protocol, measures |
| `astar.py` | A* baseline (same moves and step cost as the ants) |
| `final_experiments.py` | The 15 conditions x 10 seeds |
| `analyze.py`, `analyze2.py` | Recovery rules and the statistics in the report |
| `make_figures.py` | Report figures |
| `viewer.py` | Live demo |
| `tune.py`, `tune2.py`, `tune3.py` | Pilot runs used to choose baseline settings (seeds 100-103) |
| `results/results.jsonl` | Raw results, one line per trial |
| `figures/` | Figures used in the report |

## Method in brief

* World: a wall with two 10-cell gaps placed symmetrically about the nest-food
  line. A shelf under the food makes the lower gap's route longer. A* gives 52
  steps (short gap) and 75 (long gap); without the shelf both are 52.
* Protocol: 5,000 ticks, seal the short gap, 4,000 more ticks.
* Seeds 0-9 are for evaluation; seeds 100-103 were used only for tuning.
* "Recovered": final-1,000-tick throughput >= 0.5 x (52/75) x pre-seal throughput,
  counted only for "working" colonies (pre-seal throughput >= 3 x the no-scent
  median). The working threshold was added after seeing the no-scent result.

No participant data, credentials or API keys are used or stored.
