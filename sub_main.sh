#!/bin/bash
#$ -o qsub_output
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=40G
source /scratch_net/biwidl217/conda/etc/profile.d/conda.sh
conda activate proj 
python -u /home/jmihali/Projects/SemesterProject/PytorchNeuralStyleTransfer/main.py "$@"
