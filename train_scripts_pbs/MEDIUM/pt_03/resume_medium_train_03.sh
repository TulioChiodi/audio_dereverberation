#!/bin/bash

#PBS -N medium_03
#PBS -q umagpu
#PBS -e erros_medium_03
#PBS -o saida_medium_03

module load python/3.8.12-intel-2021.3.0
module load cuda/11.5.0-intel-2022.0.1

cd /home/lovelace/proj/proj944/a264372/audio_dereverberation
source audio_dereverb/bin/activate
cd /home/lovelace/proj/proj944/a264372/audio_dereverberation_git

mkdir -p RESULTS/MEDIUM/train_03

python src/models/train_model.py --dataset_name=medium --batch_size=21 --results_path=RESULTS/MEDIUM/train_03 --checkpoint_dir=RESULTS/MEDIUM/train_03 --load_model=RESULTS/MEDIUM/train_02/checkpoint | tee RESULTS/MEDIUM/train_03/output_03.log
