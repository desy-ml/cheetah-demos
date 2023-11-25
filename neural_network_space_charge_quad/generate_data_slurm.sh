#!/bin/sh
#SBATCH --partition=maxcpu
#SBATCH --job-name generate-data-space-charge-quad
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=75F3
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate cheetah-demos
cd /home/kaiserja/beegfs/cheetah-demos/neural_network_space_charge_quad

srun python generate_data.py 20000 data_2/test.yaml

exit
