#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name nn-space-charge-quad-sweep
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=P100|V100|A100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate cheetah-demos
cd /home/kaiserja/beegfs/cheetah-demos/neural_network_space_charge_quad

srun wandb agent --count 1 msk-ipc/space-charge-quadrupole/492rolh0

exit
