#!/bin/bash
for i in $(seq 1 5); do
  for j in $(seq 0 15); do
      export electrode_id_index=$j,mcmc_id=$i,mcmc_check_num=0,truncate_index=400
      sbatch ./script_slurm_system/Real_data_predict.pbs
  done
done
