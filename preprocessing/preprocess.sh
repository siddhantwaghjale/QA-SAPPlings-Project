#!/bin/bash
#SBATCH --job-name=rtp_2p1
#SBATCH --partition=cpu
#SBATCH --output=qa_pre1.out
#SBATCH --error=qa_pre1.err
#SBATCH --mem=196g
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyansk@andrew.cmu.edu

source /home/priyansk/.bashrc
conda activate qa
# export HF_HOME="/data/datasets/models/hf_cache/"

# HTML Processing
for i in {5000..137000..1000}
do
    python /home/priyansk/qa/MultiTabQA/preprocess_multitable_e2e.py multitable_pretraining $i $(($i+1000)) Salesforce/codet5-base html
done
