#!/bin/bash

label=$1
shift

mkdir -p policies figures benchmark_files learning_curves

python ensemble.py --exp-name ${label} --save-dir policies/${label}/policy $*
python learning_curve.py learning_curves/${label}_rewards.pkl figures/${label}_learning_curve.png
python ensemble.py --exp-name ${label}_eval --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $* --ensemble-choice episode
python statistics.py benchmark_files/${label}_eval.pkl >> ${label%_*}.csv
python prepare_ensemble.py benchmark_files/${label}_eval.pkl --policy_file policies/${label}/policy --output_file benchmark_files/${label}_eval.npz $*
python figure.py benchmark_files/${label}_eval.npz figures/${label}.png

python sheldon_ensemble.py --exp-name ${label}_sheldon12_landmark12 --num-sheldons 2 --sheldon-ids 1 2 --sheldon-targets 1 2  --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $*
python sheldon_ensemble.py --exp-name ${label}_sheldon12_landmark02 --num-sheldons 2 --sheldon-ids 1 2 --sheldon-targets 0 2  --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $*
python sheldon_ensemble.py --exp-name ${label}_sheldon12_landmark01 --num-sheldons 2 --sheldon-ids 1 2 --sheldon-targets 0 1  --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $*
python sheldon_ensemble.py --exp-name ${label}_sheldon02_landmark12 --num-sheldons 2 --sheldon-ids 0 2 --sheldon-targets 1 2  --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $*
python sheldon_ensemble.py --exp-name ${label}_sheldon02_landmark02 --num-sheldons 2 --sheldon-ids 0 2 --sheldon-targets 0 2  --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $*
python sheldon_ensemble.py --exp-name ${label}_sheldon02_landmark01 --num-sheldons 2 --sheldon-ids 0 2 --sheldon-targets 0 1  --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $*
python sheldon_ensemble.py --exp-name ${label}_sheldon01_landmark12 --num-sheldons 2 --sheldon-ids 0 1 --sheldon-targets 1 2  --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $*
python sheldon_ensemble.py --exp-name ${label}_sheldon01_landmark02 --num-sheldons 2 --sheldon-ids 0 1 --sheldon-targets 0 2  --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $*
python sheldon_ensemble.py --exp-name ${label}_sheldon01_landmark01 --num-sheldons 2 --sheldon-ids 0 1 --sheldon-targets 0 1  --load-dir policies/${label}/policy --restore --benchmark --save-replay --deterministic $*

python statistics.py benchmark_files/${label}_sheldon12_landmark12.pkl >>${label%_*}.csv
python statistics.py benchmark_files/${label}_sheldon12_landmark02.pkl >>${label%_*}.csv
python statistics.py benchmark_files/${label}_sheldon12_landmark01.pkl >>${label%_*}.csv
python statistics.py benchmark_files/${label}_sheldon02_landmark12.pkl >>${label%_*}.csv
python statistics.py benchmark_files/${label}_sheldon02_landmark02.pkl >>${label%_*}.csv
python statistics.py benchmark_files/${label}_sheldon02_landmark01.pkl >>${label%_*}.csv
python statistics.py benchmark_files/${label}_sheldon01_landmark12.pkl >>${label%_*}.csv
python statistics.py benchmark_files/${label}_sheldon01_landmark02.pkl >>${label%_*}.csv
python statistics.py benchmark_files/${label}_sheldon01_landmark01.pkl >>${label%_*}.csv
