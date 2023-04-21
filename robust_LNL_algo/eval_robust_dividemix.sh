#!/bin/sh

for r in 0.2 0.4 0.6 0.8
do
  logdir_c10="./eval_results/cifar10/preact_resnet18/r$r"
  logdir_c100="./eval_results/cifar100/preact_resnet18/r$r"
  mkdir -p $logdir_c10
  mkdir -p $logdir_c100
done

logdir_cn="./eval_results/cifarn"
logdir_clo="./eval_results/clothing"
mkdir -p $logdir_cn
mkdir -p $logdir_clo

cd LNL

# eval robust dividemix on cifar10
python robust_dividemix.py --gpu 0 --r 0.2 --dataset 'cifar10' --log 'robdm_bad' --noise-file '../eval_badlabels/noise/badlabels_cifar10_r0.2.json' --warmup 4 --lam 0.5 --tol 0.01 --max-iter 20 --noise-mode 'bad' &
python robust_dividemix.py --gpu 1 --r 0.4 --dataset 'cifar10' --log 'robdm_bad' --noise-file '../eval_badlabels/noise/badlabels_cifar10_r0.4.json' --warmup 4 --lam 0.8 --tol 0.01 --max-iter 20 --noise-mode 'bad' &
python robust_dividemix.py --gpu 2 --r 0.6 --dataset 'cifar10' --log 'robdm_bad' --noise-file '../eval_badlabels/noise/badlabels_cifar10_r0.6.json' --warmup 4 --lam 1 --tol 0.01 --max-iter 20 --noise-mode 'bad' &
python robust_dividemix.py --gpu 3 --r 0.8 --dataset 'cifar10' --log 'robdm_bad' --noise-file '../eval_badlabels/noise/badlabels_cifar10_r0.8.json' --warmup 4 --lam 1 --tol 0.01 --max-iter 20 --noise-mode 'bad' &
wait

python robust_dividemix.py --gpu 0 --r 0.2 --dataset 'cifar10' --log 'robdm_sym' --noise-file '../eval_badlabels/noise/sym_cifar10_r0.2.json' --warmup 10 --lam 0.2 --tol 0.01 --max-iter 20 --noise-mode 'sym' &
python robust_dividemix.py --gpu 1 --r 0.4 --dataset 'cifar10' --log 'robdm_sym' --noise-file '../eval_badlabels/noise/sym_cifar10_r0.4.json' --warmup 10 --lam 0.2 --tol 0.01 --max-iter 20 --noise-mode 'sym' &
python robust_dividemix.py --gpu 2 --r 0.6 --dataset 'cifar10' --log 'robdm_sym' --noise-file '../eval_badlabels/noise/sym_cifar10_r0.6.json' --warmup 10 --lam 0.2 --tol 0.01 --max-iter 20 --noise-mode 'sym' &
python robust_dividemix.py --gpu 3 --r 0.8 --dataset 'cifar10' --log 'robdm_sym' --noise-file '../eval_badlabels/noise/sym_cifar10_r0.8.json' --warmup 10 --lam 0.2 --tol 0.01 --max-iter 20 --noise-mode 'sym' &
wait

python robust_dividemix.py --gpu 0 --r 0.2 --dataset 'cifar10' --log 'robdm_idn' --noise-file '../eval_badlabels/noise/idn_cifar10_r0.2.csv' --warmup 10 --lam 0.2 --tol 0.01 --max-iter 20 --noise-mode 'idn' &
python robust_dividemix.py --gpu 1 --r 0.4 --dataset 'cifar10' --log 'robdm_idn' --noise-file '../eval_badlabels/noise/idn_cifar10_r0.4.csv' --warmup 10 --lam 0.2 --tol 0.01 --max-iter 20 --noise-mode 'idn' &
python robust_dividemix.py --gpu 2 --r 0.6 --dataset 'cifar10' --log 'robdm_idn' --noise-file '../eval_badlabels/noise/idn_cifar10_r0.6.csv' --warmup 10 --lam 0.2 --tol 0.01 --max-iter 20 --noise-mode 'idn' &
python robust_dividemix.py --gpu 3 --r 0.8 --dataset 'cifar10' --log 'robdm_idn' --noise-file '../eval_badlabels/noise/idn_cifar10_r0.8.csv' --warmup 10 --lam 0.2 --tol 0.01 --max-iter 20 --noise-mode 'idn' &
wait

python robust_dividemix.py --gpu 0 --r 0.2 --dataset 'cifar10' --log 'robdm_asym' --noise-file '../eval_badlabels/noise/asym_cifar10_r0.2.json' --warmup 10 --lam 0.2 --tol 0.001 --max-iter 10 --noise-mode 'asym' &
python robust_dividemix.py --gpu 1 --r 0.4 --dataset 'cifar10' --log 'robdm_asym' --noise-file '../eval_badlabels/noise/asym_cifar10_r0.4.json' --warmup 10 --lam 0.2 --tol 0.001 --max-iter 10 --noise-mode 'asym' &
wait

# eval robust dividemix on cifar100
python robust_dividemix.py --gpu 0 --r 0.2 --dataset 'cifar100' --log 'robdm_bad' --noise-file '../eval_badlabels/noise/badlabels_cifar100_r0.2.json' --warmup 20 --lam 1 --tol 0.01 --max-iter 50 --noise-mode 'bad' &
python robust_dividemix.py --gpu 1 --r 0.4 --dataset 'cifar100' --log 'robdm_bad' --noise-file '../eval_badlabels/noise/badlabels_cifar100_r0.4.json' --warmup 20 --lam 1 --tol 0.01 --max-iter 50 --noise-mode 'bad' &
python robust_dividemix.py --gpu 2 --r 0.6 --dataset 'cifar100' --log 'robdm_bad' --noise-file '../eval_badlabels/noise/badlabels_cifar100_r0.6.json' --warmup 10 --lam 1 --tol 0.01 --max-iter 10 --p-threshold 0.8 --perturb-threshold 0.8 --noise-mode 'bad' &
python robust_dividemix.py --gpu 3 --r 0.8 --dataset 'cifar100' --log 'robdm_bad' --noise-file '../eval_badlabels/noise/badlabels_cifar100_r0.8.json' --warmup 10 --lam 1 --tol 0.01 --max-iter 10 --p-threshold 0.8 --perturb-threshold 0.8 --noise-mode 'bad' &
wait

python robust_dividemix.py --gpu 0 --r 0.2 --dataset 'cifar100' --log 'robdm_sym' --noise-file '../eval_badlabels/noise/sym_cifar100_r0.2.json' --warmup 30 --lam 0.2 --tol 0.01 --max-iter 50 --noise-mode 'sym' &
python robust_dividemix.py --gpu 1 --r 0.4 --dataset 'cifar100' --log 'robdm_sym' --noise-file '../eval_badlabels/noise/sym_cifar100_r0.4.json' --warmup 30 --lam 0.2 --tol 0.01 --max-iter 50 --noise-mode 'sym' &
python robust_dividemix.py --gpu 2 --r 0.6 --dataset 'cifar100' --log 'robdm_sym' --noise-file '../eval_badlabels/noise/sym_cifar100_r0.6.json' --warmup 30 --lam 0.2 --tol 0.01 --max-iter 50 --noise-mode 'sym' &
python robust_dividemix.py --gpu 3 --r 0.8 --dataset 'cifar100' --log 'robdm_sym' --noise-file '../eval_badlabels/noise/sym_cifar100_r0.8.json' --warmup 30 --lam 0.2 --tol 0.01 --max-iter 50 --noise-mode 'sym' &
wait

python robust_dividemix.py --gpu 0 --r 0.2 --dataset 'cifar100' --log 'robdm_idn' --noise-file '../eval_badlabels/noise/idn_cifar100_r0.2.json' --warmup 30 --lam 0.2 --tol 0.01 --max-iter 50 --noise-mode 'idn' &
python robust_dividemix.py --gpu 1 --r 0.4 --dataset 'cifar100' --log 'robdm_idn' --noise-file '../eval_badlabels/noise/idn_cifar100_r0.4.json' --warmup 30 --lam 0.2 --tol 0.01 --max-iter 50 --noise-mode 'idn' &
python robust_dividemix.py --gpu 2 --r 0.6 --dataset 'cifar100' --log 'robdm_idn' --noise-file '../eval_badlabels/noise/idn_cifar100_r0.6.json' --warmup 30 --lam 0.2 --tol 0.01 --max-iter 50 --noise-mode 'idn' &
python robust_dividemix.py --gpu 3 --r 0.8 --dataset 'cifar100' --log 'robdm_idn' --noise-file '../eval_badlabels/noise/idn_cifar100_r0.8.json' --warmup 30 --lam 0.2 --tol 0.01 --max-iter 50 --noise-mode 'idn' &
wait

# eval robust dividemix on cifarn
python robust_dividemix_cifarn.py --gpu 0 --log 'robdm_cifarn_worst' --warmup 10 --lam 0.2 &
wait

# eval robust dividemix on clothing1m
python robust_dividemix_clothing.py --gpu 0 --log 'robdm_clothing1m' --warmup 2 --lam 0.2 &
wait
