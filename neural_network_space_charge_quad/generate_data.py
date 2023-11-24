import argparse
import multiprocessing
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import ocelot
import yaml
from loky import get_reusable_executor
from ocelot.cpbd.beam import generate_parray


def generate_sample(idx: int, target_dir: str) -> None:
    """
    Generate a sample of tracking through a quadrupole magnet in Ocelot. It saves
    incoming and outgoing beam parameters as well as controls.

    :param idx: Unique index of the sample.
    :param target_dir: Directory where to save the generated data set.
    """
    # Generate control values
    length = np.random.uniform(0.05, 0.5)
    k1 = np.random.uniform(-72.0, 72.0)

    # Create Ocelot cell
    quadrupole = ocelot.Quadrupole(l=length, k1=k1)
    cell = [quadrupole]

    # Create Ocelot beam
    p_array_incoming = generate_parray(
        sigma_x=np.random.uniform(1e-5, 1e-3),
        sigma_px=np.random.uniform(1e-5, 1e-3),
        sigma_y=np.random.uniform(1e-5, 1e-3),
        sigma_py=np.random.uniform(1e-5, 1e-3),
        sigma_tau=np.random.uniform(3e-7, 3e-4),
        sigma_p=np.random.uniform(1e-5, 1e-3),
        charge=np.random.uniform(1e-12, 5e-9),
        nparticles=np.random.randint(100_000),
        energy=np.random.uniform(0.001, 1.0),  # GeV
    )

    # Run tracking
    method = {"global": ocelot.SecondTM}
    lattice = ocelot.MagneticLattice(cell, method=method)

    space_charge = ocelot.SpaceCharge()
    space_charge.nmesh_xyz = [63, 63, 63]
    space_charge.step = 1

    navigator = ocelot.Navigator(lattice)
    navigator.add_physics_proc(space_charge, quadrupole, quadrupole)
    navigator.unit_step = 0.02

    p_array = deepcopy(p_array_incoming)
    _, p_array_outgoing = ocelot.track(lattice, p_array, navigator)

    # Save incoming and outgoing beam parameters as well as controls
    dataset_dir = Path(target_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / f"{idx:09d}.yaml", "w") as f:
        sample_dict = {
            "controls": {"length": length, "k1": k1},
            "incoming": {
                "sigma_x": p_array_incoming.x().std().item(),
                "sigma_px": p_array_incoming.px().std().item(),
                "sigma_y": p_array_incoming.y().std().item(),
                "sigma_py": p_array_incoming.py().std().item(),
                "sigma_tau": p_array_incoming.tau().std().item(),
                "sigma_p": p_array_incoming.p().std().item(),
                "charge": p_array_incoming.total_charge.item(),
                "energy": p_array_incoming.E * 1e9,  # Convert from GeV to eV
                "nparticles": p_array_incoming.size(),
            },
            "outgoing": {
                "sigma_x": p_array_outgoing.x().std().item(),
                "sigma_px": p_array_outgoing.px().std().item(),
                "sigma_y": p_array_outgoing.y().std().item(),
                "sigma_py": p_array_outgoing.py().std().item(),
                "sigma_tau": p_array_outgoing.tau().std().item(),
                "sigma_p": p_array_outgoing.p().std().item(),
                "charge": p_array_outgoing.total_charge.item(),
                "energy": p_array_outgoing.E * 1e9,  # Convert from GeV to eV
                "nparticles": p_array_outgoing.size(),
            },
        }
        yaml.dump(sample_dict, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("num_samples", type=int, help="Number of samples to generate.")
    parser.add_argument(
        "target_dir", type=str, help="Directory where to save the generated data set."
    )
    args = parser.parse_args()

    generate_sample_to_target_dir = partial(generate_sample, target_dir=args.target_dir)

    # for idx in range(args.num_samples):
    #     generate_sample_to_target_dir(idx)

    executor = get_reusable_executor(max_workers=multiprocessing.cpu_count())
    executor.map(generate_sample_to_target_dir, range(args.num_samples), chunksize=100)


if __name__ == "__main__":
    main()
