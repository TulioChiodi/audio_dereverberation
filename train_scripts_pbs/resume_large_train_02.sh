#!/bin/bash

#PBS -N large_02
#PBS -q umagpu
#PBS -e erros_large_02
#PBS -o saida_large_02

module load python/3.8.12-intel-2021.3.0
module load cuda/11.5.0-intel-2022.0.1

cd /home/lovelace/proj/proj944/a264372/audio_dereverberation
source audio_dereverb/bin/activate
cd /home/lovelace/proj/proj944/a264372/audio_dereverberation_git

mkdir -p RESULTS/LARGE/train_02

python src/models/train_model.py --dataset_name=large --batch_size=21 --results_path=RESULTS/LARGE/train_02 --checkpoint_dir=RESULTS/LARGE/train_02 --load_model=RESULTS/LARGE/train_00/checkpoint | tee RESULTS/LARGE/train_02/output_02.log
