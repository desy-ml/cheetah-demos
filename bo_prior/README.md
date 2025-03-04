# Demonstration BO Prior Mean

This part contains a simple demonstration on using Cheetah simulation as an informed prior mean module for Bayesian optimization (BO).

Here we use the BO implementation from the Xopt package.

## Folder Structure

- `bo_cheetah_prior.py` Single file implementation of the example
  - `simple_fodo_problem` Wraps around a FODO focusing task to be conformed with the Xopt requirements.
  - `FODOPriorMean` is the mean module for BO. Note: the forward speed is currently slow due to the non-batched evaluation. (This is planned to be improved in future Cheetah versions.)
- `eval_fodo.py` runs evaluation with different conditions. See below.
- `data/` stores the Xopt run results (`pd.Dataframe`) as CSV files.
- `plot_bo_cheetah_result.ipynb` generates the result plot.

## Running Evaluations

Possible arguments:

- Switch to `--task=mismatched` to run evaluation for mismatched lattice.
- Use `-n` to change the number of trials to run for each configuration, by default `-n 10` is used.
- Use `-o path_to_dir` to specify the output folder, by default the runs will be saved ot `data/` folder

### Nelder-Mead Simplex

```bash
python eval_fodo.py --optimizer=NM -s=50 --task=matched_prior_newtask
```

### Normal BO with UCB and Zero Mean

```bash
python eval_fodo.py --optimizer=BO -s=50 --task=matched_prior_newtask
```

### BO with Cheetah Model as Prior Mean

```bash
python eval_fodo.py --optimizer=BO_prior -s=50 --task=matched_prior_newtask
```
