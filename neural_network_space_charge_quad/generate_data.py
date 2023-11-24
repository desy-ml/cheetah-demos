import argparse
import multiprocessing
from copy import deepcopy
from pathlib import Path

import numpy as np
import ocelot
import yaml
from loky import get_reusable_executor
from ocelot.cpbd.beam import generate_parray


def generate_sample(idx: int) -> None:
    """
    Generate a sample of tracking through a quadrupole magnet in Ocelot. It saves
    incoming and outgoing beam parameters as well as controls.

    :param idx: Unique index of the sample.
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
    mu_x = np.random.uniform(-1e-3, 1e-3)
    mu_xp = np.random.uniform(-1e-6, 1e-6)
    mu_y = np.random.uniform(-1e-3, 1e-3)
    mu_yp = np.random.uniform(-1e-6, 1e-6)
    p_array_incoming.rparticles[0] += mu_x
    p_array_incoming.rparticles[1] += mu_xp
    p_array_incoming.rparticles[2] += mu_y
    p_array_incoming.rparticles[3] += mu_yp

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
    sample_dir = Path(f"data/{idx:09d}")
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "controls.yaml", "w") as f:
        controls_dict = {"length": length, "k1": k1}
        yaml.dump(controls_dict, f)
    ocelot.save_particle_array(sample_dir / "incoming.npz", p_array_incoming)
    ocelot.save_particle_array(sample_dir / "outgoing.npz", p_array_outgoing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("numsamples", type=int, help="Number of samples to generate.")
    args = parser.parse_args()

    executor = get_reusable_executor(max_workers=multiprocessing.cpu_count())
    executor.map(generate_sample, range(args.numsamples))


if __name__ == "__main__":
    main()
