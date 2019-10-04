#!/bin/sh
#$ -cwd
#$ -l long
#$ -l vf=4G
#$ -pe smp 32
#$ -t 1
#$ -e ./logs/
#$ -o ./logs/
. /data/nlp/lunar_pilot_env/bin/activate

mkdir -p ./models/
mkdir -p ./runs/

python main_mp.py --num_agents 32
