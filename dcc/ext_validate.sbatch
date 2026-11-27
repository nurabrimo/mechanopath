#!/bin/bash
#SBATCH -J mp_ext
#SBATCH -p scavenger-gpu,gpu-common
#SBATCH --gres=gpu:1
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-23,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-carlsonlab-gpu-ferc-s-o15-10,dcc-chsi-gpu-ferc-s-i11-1,dcc-core-ferc-s-z25-20,dcc-core-ferc-s-z25-21,dcc-core-gpu-ferc-s-h36-5,dcc-core-gpu-ferc-s-h36-6,dcc-core-gpu-ferc-s-h36-9,dcc-core-gpu-ferc-s-p15-20,dcc-gehmlab-gpu-ferc-s-n32-10,dcc-gehmlab-gpu-ferc-s-n32-11,dcc-gehmlab-gpu-ferc-s-n32-12,dcc-gehmlab-gpu-ferc-s-n32-13,dcc-gehmlab-gpu-ferc-s-n32-23,dcc-gehmlab-gpu-ferc-s-n32-24,dcc-gehmlab-gpu-ferc-s-z25-15,dcc-gehmlab-gpu-ferc-s-z25-17,dcc-gehmlab-gpu-ferc-s-z25-18,dcc-gehmlab-gpu-ferc-s-z25-19,dcc-pearsonlab-gpu-ferc-s-o15-17,dcc-plusds-gpu-ferc-s-j11-18,dcc-plusds-gpu-ferc-s-z25-23,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-gehmlab-gpu-ferc-s-z25-18,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --requeue
#SBATCH -o /hpc/home/nb295/mechanopath_full/logs/mp_ext_%j.out
source ~/mechanopath_full/dcc/config.sh
$CONDA_ENV/bin/python ~/mechanopath_full/dcc/ext_validate.py
