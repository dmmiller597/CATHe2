#!/bin/bash

# Train ProFam

#$ -l tmem=40G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=12:55:30
#$ -S /bin/bash
#$ -N contrasTED
#$ -o /SAN/orengolab/functional-families/CATHe2/qsub_logs/
#$ -wd /SAN/orengolab/functional-families/CATHe2/
#$ -P cath
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate venvPF
ROOT_DIR='/SAN/orengolab/functional-families/CATHe2'
cd $ROOT_DIR
export WANDB__SERVICE_WAIT=300
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/src/contrastive_train.py
date