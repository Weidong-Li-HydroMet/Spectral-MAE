#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH -o run_SL_vs_SSL_RMSE.out
#SBATCH -e run_SL_vs_SSL_RMSE.err

#SBATCH --account=145439188689

cd /home/u.dp200518/SSL-Wavelets/eval

source activate /scratch/user/u.dp200518/.conda/envs/2DTurbEmulator_v1

# python -u SL_vs_SSL_RMSE.py <config_file_name> </path/to/ckpot_root> <SSL_run_num>
python -u SL_vs_SSL_RMSE.py config_video.yaml /scratch/user/u.dp200518/SSL-2DTurb/MAE_FINETUNE/ 0001
