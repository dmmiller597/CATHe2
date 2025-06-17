#!/bin/bash

# Embed sequences using ProtT5

#$ -l tmem=80G
#$ -l gpu=true
#$ -l gpu_type=a100_80
#$ -l h_rt=24:00:00
#$ -S /bin/bash
#$ -N embeddings
#$ -P cath
#$ -o /SAN/orengolab/functional-families/CATHe2/qsub_logs/
#$ -wd /SAN/orengolab/functional-families/CATHe2/
#$ -j y

# Exit immediately if a command exits with a non-zero status.
set -e

echo "#################### JOB DETAILS ####################"
date
hostname
echo "Job ID: ${JOB_ID}"
echo "Job Name: ${JOB_NAME}"
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
ROOT_DIR='/SAN/orengolab/functional-families/CATHe2'
cd $ROOT_DIR
source ${ROOT_DIR}/venv/bin/activate
export HF_HOME=${ROOT_DIR}/model_cache
echo 'starting python script'
python ${ROOT_DIR}/scripts/embed_prott5.py --max-tokens 200000
echo 'python script finished'
date