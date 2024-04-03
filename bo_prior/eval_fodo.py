"""Evaluate the FODO optimization task using different optimizers.

task modes:
    - matched: Optimize the FODO lattice (*) for a matched beam.
    - mismatched: Optimize a new FODO lattice for a mismatched prior (for *).
    - matched_prior_newtask: Optimize the new FODO lattice for a matched with a matched priror.
"""

import os

import bo_cheetah_prior
import cheetah
import pandas as pd
import torch
import tqdm
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.scipy.neldermead import NelderMeadGenerator


def main(args):
    # VOCS
    vocs_config = """ 
        variables:
            q1: [-30, 15]
            q2: [-15, 30]
        objectives:
            mae: minimize
    """
    vocs = VOCS.from_yaml(vocs_config)

    # Evaluator
    if args.task == "matched":
        incoming_beam = None
        evaluator = Evaluator(
            function=bo_cheetah_prior.simple_fodo_problem,
            function_kwargs={"incoming_beam": incoming_beam},
        )
    elif args.task == "mismatched":
        # Both incoming beam and the lattice distance are incorrect
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor([1e-3]),
            sigma_y=torch.tensor([1e-3]),
            sigma_xp=torch.tensor([1e-4]),
            sigma_yp=torch.tensor([1e-4]),
            energy=torch.tensor([100e6]),
        )
        evaluator = Evaluator(
            function=bo_cheetah_prior.simple_fodo_problem,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "lattice_distances": {"drift_length": 0.7},
            },
        )
    elif args.task == "matched_prior_newtask":
        # Lattice distance
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor([1e-3]),
            sigma_y=torch.tensor([1e-3]),
            sigma_xp=torch.tensor([1e-4]),
            sigma_yp=torch.tensor([1e-4]),
            energy=torch.tensor([100e6]),
        )
        evaluator = Evaluator(
            function=bo_cheetah_prior.simple_fodo_problem,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "lattice_distances": {"drift_length": 0.7},
            },
        )

    # Empty dataframe to store results
    df = pd.DataFrame()

    # Run n_trials
    for i in range(args.n_trials):
        print(f"Trial {i+1}/{args.n_trials}")

        # Initialize Generator
        if args.optimizer == "BO":
            generator = UpperConfidenceBoundGenerator(beta=2.0, vocs=vocs)
        elif args.optimizer == "BO_prior":
            prior_mean_module = bo_cheetah_prior.FodoPriorMean()
            prior_mean_module.drift_length = 0.5
            if args.task == "matched":
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
            elif args.task == "mismatched":
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                    trainable_mean_keys=["mae"],  # Allow the prior mean to be trained
                )
            elif args.task == "matched_prior_newtask":
                incoming_beam = cheetah.ParameterBeam.from_parameters(
                    sigma_x=torch.tensor([1e-3]),
                    sigma_y=torch.tensor([1e-3]),
                    sigma_xp=torch.tensor([1e-4]),
                    sigma_yp=torch.tensor([1e-4]),
                    energy=torch.tensor([100e6]),
                )
                prior_mean_module = bo_cheetah_prior.FodoPriorMean(
                    incoming_beam=incoming_beam
                )
                prior_mean_module.drift_length = 0.7
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
            generator = UpperConfidenceBoundGenerator(
                beta=2.0, vocs=vocs, gp_constructor=gp_constructor
            )
        elif args.optimizer == "NM":
            generator = NelderMeadGenerator(vocs=vocs)
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")

        xopt = Xopt(
            vocs=vocs,
            evaluator=evaluator,
            generator=generator,
            max_evaluations=args.max_evaluation_steps,
        )
        # Fixed starting point
        xopt.evaluate_data(
            {
                "q1": -20.0,
                "q2": 20.0,
            }
        )
        # Start Optimization
        for _ in tqdm.tqdm(range(args.max_evaluation_steps)):
            xopt.step()

        # xopt.run()
        # Post processing the dataframes
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])
    # Check if the outfile directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}.csv"

    df.to_csv(out_filename)


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Run FODO optimization task.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="BO",
        choices=["BO", "BO_prior", "NM"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--n_trials",
        "-n",
        type=int,
        default=10,
        help="Number of trials to run for each optimizer.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="matched",
        choices=["matched", "mismatched", "matched_prior_newtask"],
        help="Task to run. See bo_cheetah_prior.py for options.",
    )
    parser.add_argument(
        "--max_evaluation_steps",
        "-s",
        type=int,
        default=20,
        help="Maximum number of evaluations to run for each trial.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/",
        help="Output file path for results.",
    )
    parser.add_argument(
        "--n_workers",
        "-w",
        type=int,
        default=mp.cpu_count() - 1,
        help="Number of workers to use for parallel evaluation.",
    )
    args = parser.parse_args()
    main(args)
