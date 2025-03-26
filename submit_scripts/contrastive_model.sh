#!/bin/bash

# Train contrastive model    

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=a100_80
#$ -l h_rt=24:00:00
#$ -S /bin/bash
#$ -N contrastive_model
#$ -P cath
#$ -o /SAN/orengolab/functional-families/CATHe2/qsub_logs/
#$ -wd /SAN/orengolab/functional-families/CATHe2/
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate venv
ROOT_DIR='/SAN/orengolab/functional-families/CATHe2'
cd $ROOT_DIR
export WANDB__SERVICE_WAIT=900
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/src/contrastive_train.py
date