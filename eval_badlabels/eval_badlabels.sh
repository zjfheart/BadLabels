#!/bin/sh

r=0.2
dataset='cifar10'
net='preact_resnet18'
dpath='../../data'
sym="./noise/sym_"$dataset"_r"$r".json"
asym="./noise/asym_"$dataset"_r"$r".json"
idn="./noise/idn_"$dataset"_r"$r".csv"
bad="./noise/badlabels_"$dataset"_r"$r".json"
logdir="./eval_results/$dataset/$net/r$r"

mkdir -p $logdir
cd LNL

# hyper-parameters for dividemix
lambda_u=0  # 0 for cifar10(r0.2, r0.4) and mnist(r0.2, r0.4), 25 for cifar10(r0.6, r0.8) and mnist(r0.6, r0.8) and cifar100(r0.2, r0.4), 150 for cifar100(r0.6, r0.8)
p_threshold=0.5
# hyper-parameters for pgdf
preset=c10.lu0  # c10.lu0 for cifar10(r0.2, r0.4), c10.lu25 for cifar10(r0.6, r0.8), c100.lu25 for cifar100(r0.2, r0.4), c100.lu150 for cifar100(r0.6, r0.8)
# hyper-parameters for sop
config='./config/sop_config_c10.json'  # './config/sop_config_c10.json' for cifar10, './config/sop_config_c100.json' for cifar100, './config/sop_config_mnist.json' for mnist
# hyper-parameters for adacorr
ep_start=25  # 25 for cifar10(r0.2, r0.4, r0.6) and cifar100(r0.2, r0.4, r0.6), 20 for cifar10(r0.8), 30 for cifar100(r0.8)
ep_update=30  # 30 for cifar10(r0.2, r0.4, r0.6) and cifar100(r0.2, r0.4, r0.6), 25 for cifar10(r0.8), 35 for cifar100(r0.8)

# standard training
python standard_training.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --r $r --log 'st_sym' --noise-file $sym &
python standard_training.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --r $r --log 'st_asym' --noise-mode 'asym' --noise-file $asym &
python standard_training.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --r $r --log 'st_idn' --noise-file $idn &
python standard_training.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --r $r --log 'st_bad' --noise-file $bad &
wait

# co-teaching
python coteaching.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --noise-rate $r --log 'ct_sym' --noise-file $sym &
python coteaching.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --noise-rate $r --log 'ct_asym' --noise-type 'asym' --noise-file $asym &
python coteaching.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --noise-rate $r --log 'ct_idn' --noise-file $idn &
python coteaching.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --noise-rate $r --log 'ct_bad' --noise-file $bad &
wait

# t_revision
python t_revision.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --noise-rate $r --log 'tr_sym' --noise-file $sym &
python t_revision.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --noise-rate $r --log 'tr_asym' --noise-mode 'asym' --noise-file $asym &
python t_revision.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --noise-rate $r --log 'tr_idn' --noise-file $idn &
python t_revision.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --noise-rate $r --log 'tr_bad' --noise-file $bad &
wait

# rog
python rog.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --noise-fraction $r --log 'rog_sym' --noise-file $sym &
python rog.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --noise-fraction $r --log 'rog_asym' --noise-type 'asym' --noise-file $asym &
python rog.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --noise-fraction $r --log 'rog_idn' --noise-file $idn &
python rog.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --noise-fraction $r --log 'rog_bad' --noise-file $bad &
wait

# dividemix (lambda-u, p-threshold)
python dividemix.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --r $r --lambda-u $lambda_u --p-threshold $p_threshold --log 'dm_sym' --noise-file $sym &
python dividemix.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --r $r --lambda-u $lambda_u --p-threshold $p_threshold --log 'dm_asym' --noise-mode 'asym' --noise-file $asym &
python dividemix.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --r $r --lambda-u $lambda_u --p-threshold $p_threshold --log 'dm_idn' --noise-file $idn &
python dividemix.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --r $r --lambda-u $lambda_u --p-threshold $p_threshold --log 'dm_bad' --noise-file $bad &
wait

# adacorr (epoch-start, epoch-update)
python ada_corr.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --noise-level $r --epoch-start $ep_start --epoch-update $ep_update --log 'ac_sym' &
python ada_corr.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --noise-level $r --epoch-start $ep_start --epoch-update $ep_update --log 'ac_asym' --noise-type 'asym' &
python ada_corr.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --noise-level $r --epoch-start $ep_start --epoch-update $ep_update --log 'ac_idn' --noise-file $idn &
python ada_corr.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --noise-level $r --epoch-start $ep_start --epoch-update $ep_update --log 'ac_bad' --noise-file $bad &
wait

# peer loss
python peer_loss.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --r $r --log 'pl_sym' --noise-file $sym &
python peer_loss.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --r $r --log 'pl_asym' --noise-mode 'asym' --noise-file $asym &
python peer_loss.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --r $r --log 'pl_idn' --noise-file $idn &
python peer_loss.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --r $r --log 'pl_bad' --noise-file $bad &
wait

# negative label smoothing (smooth-rate default:-1.0)
python negative_ls.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --noise-rate $r --log 'nls_sym' --noise-file $sym &
python negative_ls.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --noise-rate $r --log 'nls_asym' --noise-type 'asym' --noise-file $asym &
python negative_ls.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --noise-rate $r --log 'nls_idn' --noise-file $idn &
python negative_ls.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --noise-rate $r --log 'nls_bad' --noise-file $bad &
wait

# pgdf get prior (preset)
python pgdf_getprior.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --r $r --preset $preset --log 'pgdf_sym' --noise-file $sym &
python pgdf_getprior.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --r $r --preset $preset --log 'pgdf_asym' --noise-mode 'asym' --noise-file $asym &
python pgdf_getprior.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --r $r --preset $preset --log 'pgdf_idn' --noise-file $idn &
python pgdf_getprior.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --r $r --preset $preset --log 'pgdf_bad' --noise-file $bad &
wait

# pgdf (preset)
python pgdf.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --r $r --preset $preset --log 'pgdf_sym' --noise-file $sym &
python pgdf.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --r $r --preset $preset --log 'pgdf_asym' --noise-mode 'asym' --noise-file $sym &
python pgdf.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --r $r --preset $preset --log 'pgdf_idn' --noise-file $idn &
python pgdf.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --r $r --preset $preset --log 'pgdf_bad' --noise-file $bad &
wait

# promix
python promix.py --dataset $dataset --data-path $dpath --net $net --gpu 0 --r $r --log 'pm_sym' --noise-file $sym &
python promix.py --dataset $dataset --data-path $dpath --net $net --gpu 1 --r $r --log 'pm_asym' --noise-type 'asym' --noise-file $asym &
python promix.py --dataset $dataset --data-path $dpath --net $net --gpu 2 --r $r --log 'pm_idn' --noise-file $idn &
python promix.py --dataset $dataset --data-path $dpath --net $net --gpu 3 --r $r --log 'pm_bad' --noise-file $bad &
wait

# sop (c)
python sop.py -d 0 -c $config --data_dir $dpath --net $net --percent $r --log 'sop_sym' --noise_file $sym &
python sop.py -d 1 -c $config --data_dir $dpath --net $net --percent $r --log 'sop_asym' --asym True --noise_file $asym &
python sop.py -d 2 -c $config --data_dir $dpath --net $net --percent $r --log 'sop_idn' --instance True --noise_file $idn &
python sop.py -d 3 -c $config --data_dir $dpath --net $net --percent $r --log 'sop_bad' --att True --noise_file $bad &
wait
