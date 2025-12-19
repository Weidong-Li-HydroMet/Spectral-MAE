#!/bin/bash -l                                                          
#SBATCH --time=05:00:00                                                     
#SBATCH -C 'gpu&hbm40g'
#SBATCH --account=m4416                                         
#SBATCH -q regular                                                      
#SBATCH --nodes=1                                                      
#SBATCH --ntasks-per-node=1                                             
#SBATCH --gpus-per-node=1                                               
#SBATCH --cpus-per-task=4
#SBATCH -J analyze_model
#SBATCH --module=gpu,nccl-2.18                                          
#SBATCH -o /pscratch/sd/d/dpp94/Logfiles/analyze_model.out
#SBATCH --mail-type=begin,end,fail                                      
#SBATCH --mail-user=dpp94@uchicago.edu 

cd /global/homes/d/dpp94/SSL-for-2D-Turbulence/eval

ml conda
conda activate 2DTurbEmulator

python -u analyze_model.py config_video_v1_PM.yaml
