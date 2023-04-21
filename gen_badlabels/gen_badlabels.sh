#!/bin/sh

# gen cifar-10 badlabels
python badlabels.py --dataset cifar10 --flip-budget 0.2 --gpu 0 --gen &
wait
python badlabels.py --dataset cifar10 --flip-budget 0.4 --gpu 0 --gen --z-load './z/z_cifar10_e120.npy' &
wait
python badlabels.py --dataset cifar10 --flip-budget 0.6 --gpu 0 --gen --z-load './z/z_cifar10_e120.npy' &
wait
python badlabels.py --dataset cifar10 --flip-budget 0.8 --gpu 0 --gen --z-load './z/z_cifar10_e120.npy' &
wait

# gen cifar-100 badlabels
python badlabels.py --dataset cifar100 --flip-budget 0.2 --gpu 0 --gen &
wait
python badlabels.py --dataset cifar100 --flip-budget 0.4 --gpu 0 --gen --z-load './z/z_cifar100_e120.npy' &
wait
python badlabels.py --dataset cifar100 --flip-budget 0.6 --gpu 0 --gen --z-load './z/z_cifar100_e120.npy' &
wait
python badlabels.py --dataset cifar100 --flip-budget 0.8 --gpu 0 --gen --z-load './z/z_cifar100_e120.npy' &
wait

# gen mnist badlabels
python badlabels.py --dataset mnist --epochs 20 --lr 0.01 --momentum 0.5 --weight-decay 0 --flip-budget 0.2 --gpu 0 --gen &
wait
python badlabels.py --dataset mnist --flip-budget 0.4 --gpu 0 --gen --z-load './z/z_mnist_e20.npy' &
wait
python badlabels.py --dataset mnist --flip-budget 0.6 --gpu 0 --gen --z-load './z/z_mnist_e20.npy' &
wait
python badlabels.py --dataset mnist --flip-budget 0.8 --gpu 0 --gen --z-load './z/z_mnist_e20.npy' &
wait
