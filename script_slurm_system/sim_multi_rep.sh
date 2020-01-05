#!/bin/bash
for i in $(seq 1 24); do
    export sim_name_id=$i
    sbatch ./script_slurm_system/sim_multi_rep_direct.pbs
    sbatch ./script_slurm_system/sim_multi_rep_convol.pbs
done
