# Data-Driven Discovery of Dynamics in Evolutionary Game Theory

Code for the master's thesis. It applies SINDy to the Hawk-Dove game and recovers the
macroscopic governing equation from trajectory data. Three datasets are used: synthetic
data generated from the replicator equation, and genetic-algorithm data under two death
rules (inverse-proportional and uniform).

## Files

`genetic_algorithm.py`
Simulates the Hawk-Dove game with a genetic algorithm and saves the trajectories as CSV
files (a training and a held-out set). The death rule is chosen with the `death_rule`
variable at the top, either `"inverse"` or `"uniform"`.

`SINDy_Synthetic.py`
Runs the discovery pipeline on the synthetic data over a range of noise levels and writes
the results table and the figures.

`SINDy_GA.py`
Runs the discovery pipeline on the genetic-algorithm data over a range of ensemble
averages. The dataset (inverse or uniform death) is chosen with the `death_rule` variable
at the top, which has to match the one used in `genetic_algorithm.py`.

## How to run

1. Run `genetic_algorithm.py` twice, once with `death_rule = "inverse"` and once with
   `"uniform"`, to generate the two GA datasets.
2. Run `SINDy_Synthetic.py` for the synthetic results.
3. Run `SINDy_GA.py` for the GA results, once per death rule.

The genetic algorithm runs in parallel and takes a while. Each script writes its output to
a `Results` folder in the working directory (set by `RESULTS_ROOT` at the top of the script).

## Requirements

Python 3 with numpy, pandas, scipy, matplotlib, pysindy and openpyxl.

