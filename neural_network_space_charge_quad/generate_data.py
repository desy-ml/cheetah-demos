import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import cheetah
import numpy as np
import ocelot
import torch
import yaml
from ocelot.cpbd.beam import generate_parray
from retry import retry
from tqdm import tqdm

# NOTE: This script should be run with `export OMP_NUM_THREADS=1`

np.seterr(all="raise")


def track_ocelot() -> Tuple[ocelot.ParticleArray, float, float, ocelot.ParticleArray]:
    """
    Generate beam in Ocelot and control settings, then track the beam through a
    quadrupole magnet in Ocelot.

    :return: Tuple of incoming beam, length, k1 and outgoing beam.
    """
    # Generate control values
    length = np.random.uniform(0.05, 0.5)
    k1 = np.random.uniform(-72.0, 72.0)

    # Create Ocelot cell
    quadrupole = ocelot.Quadrupole(l=length, k1=k1)
    marker = ocelot.Marker("dummy")  # This marker is needed for space charge to work
    cell = [quadrupole, marker]

    # Create Ocelot beam
    p_array_incoming = generate_parray(
        sigma_x=np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3))),
        sigma_px=np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3))),
        sigma_y=np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3))),
        sigma_py=np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3))),
        sigma_tau=np.exp(np.random.uniform(np.log(3e-7), np.log(3e-4))),
        sigma_p=np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3))),
        charge=np.exp(np.random.uniform(np.log(1e-12), np.log(5e-9))),
        nparticles=100_000,
        energy=np.exp(np.random.uniform(np.log(0.001), np.log(1.0))),  # GeV
    )

    # Run tracking
    method = {"global": ocelot.SecondTM}
    lattice = ocelot.MagneticLattice(cell, method=method)

    space_charge = ocelot.SpaceCharge()
    space_charge.nmesh_xyz = [63, 63, 63]
    space_charge.step = 1

    navigator = ocelot.Navigator(lattice)
    navigator.add_physics_proc(space_charge, cell[0], cell[-1])
    navigator.unit_step = 0.02

    p_array = deepcopy(p_array_incoming)
    _, p_array_outgoing = ocelot.track(lattice, p_array, navigator)

    return p_array_incoming, length, k1, p_array_outgoing


def compute_ocelot_cheetah_delta(
    incoming_p_array: ocelot.ParticleArray,
    length: float,
    k1: float,
    outgoing_p_array: ocelot.ParticleArray,
) -> Tuple[cheetah.ParameterBeam, dict]:
    """
    Compute the difference between the Ocelot and Cheetah beam parameters.

    :param incoming: Incoming beam parameters.
    :param length: Length of the quadrupole magnet.
    :param k1: k1 of the quadrupole magnet.
    :param outgoing: Outgoing beam parameters.
    :return:Tuple of incoming beam converted to Cheetah and a dictionary of differences
        denoting how the Cheetah beam would have to be changed to match the Ocelot beam
        considering space charge for the parameters sigma_x, sigma_xp, sigma_y,
        sigma_yp, sigma_s, sigma_p.
    """
    incoming = cheetah.ParameterBeam.from_ocelot(incoming_p_array)
    outgoing_ocelot = cheetah.ParameterBeam.from_ocelot(outgoing_p_array)

    quadrupole = cheetah.Quadrupole(
        length=torch.tensor(length, dtype=torch.float32),
        k1=torch.tensor(k1, dtype=torch.float32),
    )

    outgoing_cheetah = quadrupole.track(incoming)

    # Compute differences
    outgoing_deltas = {
        "sigma_x": (outgoing_ocelot.sigma_x - outgoing_cheetah.sigma_x).item(),
        "sigma_xp": (outgoing_ocelot.sigma_xp - outgoing_cheetah.sigma_xp).item(),
        "sigma_y": (outgoing_ocelot.sigma_y - outgoing_cheetah.sigma_y).item(),
        "sigma_yp": (outgoing_ocelot.sigma_yp - outgoing_cheetah.sigma_yp).item(),
        "sigma_s": (outgoing_ocelot.sigma_s - outgoing_cheetah.sigma_s).item(),
        "sigma_p": (outgoing_ocelot.sigma_p - outgoing_cheetah.sigma_p).item(),
    }

    return incoming, outgoing_deltas


@retry((ValueError, FloatingPointError, ValueError), tries=100)
def generate_sample() -> Dict:
    """
    Generate a sample of tracking through a quadrupole magnet in Ocelot. It saves
    incoming and outgoing beam parameters as well as controls.
    """

    np.random.seed(None)  # Workaround for Ocelot abusing NumPy's global random state

    # Track beam in Ocelot
    p_array_incoming, length, k1, p_array_outgoing = track_ocelot()

    # Compute differences between Ocelot and Cheetah
    incoming_cheetah, outgoing_deltas = compute_ocelot_cheetah_delta(
        p_array_incoming, length, k1, p_array_outgoing
    )

    # Ensure we only get positive outgoing deltas
    if any(delta <= 0 for delta in outgoing_deltas.values()):
        raise ValueError("Negative outgoing deltas")

    # Return dictionary of incoming and outgoing beam parameters as well as controls
    return {
        "incoming": {
            "sigma_x": incoming_cheetah.sigma_x.item(),
            "sigma_xp": incoming_cheetah.sigma_xp.item(),
            "sigma_y": incoming_cheetah.sigma_y.item(),
            "sigma_yp": incoming_cheetah.sigma_yp.item(),
            "sigma_s": incoming_cheetah.sigma_s.item(),
            "sigma_p": incoming_cheetah.sigma_p.item(),
            "total_charge": incoming_cheetah.total_charge.item(),
            "energy": incoming_cheetah.energy.item(),
        },
        "controls": {"length": length, "k1": k1},
        "outgoing_deltas": outgoing_deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("num_samples", type=int, help="Number of samples to generate.")
    parser.add_argument(
        "target_file", type=str, help="File where to save the generated data set."
    )
    args = parser.parse_args()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_sample) for _ in range(args.num_samples)]
        results = [
            future.result()
            for future in tqdm(as_completed(futures), total=len(futures))
        ]

    # Save results to YAML file
    target_file = Path(args.target_file)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, "w") as f:
        yaml.dump(results, f)


if __name__ == "__main__":
    main()
