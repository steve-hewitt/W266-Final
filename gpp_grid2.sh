#!/usr/bin/env bash
python3 T5_SQuAD_QG_With_GPP.py --task ft --flag grid_e2 --encoder_max_length 64 --epochs 2 --batch_size 32 --kernel_v 100 --kernel_r 0.001 --num_return_sequences 1 --ckpt 1200 --timestamp 2022-07-21-10-49-07
python3 T5_SQuAD_QG_With_GPP.py --task ft --flag grid_e3 --encoder_max_length 64 --epochs 2 --batch_size 32 --kernel_v 100 --kernel_r 0.01 --num_return_sequences 1 --ckpt 1200 --timestamp 2022-07-21-12-14-22
python3 T5_SQuAD_QG_With_GPP.py --task ft --flag grid_e4 --encoder_max_length 64 --epochs 2 --batch_size 32 --kernel_v 100 --kernel_r 0.1 --num_return_sequences 1 --ckpt 1200 --timestamp 2022-07-21-13-43-28
python3 T5_SQuAD_QG_With_GPP.py --task ft --flag grid_d5 --encoder_max_length 64 --epochs 2 --batch_size 32 --kernel_v 10 --kernel_r 1 --num_return_sequences 1 --ckpt 1200 --timestamp 2022-07-21-06-31-52
python3 T5_SQuAD_QG_With_GPP.py --task ft --flag grid_c5 --encoder_max_length 64 --epochs 2 --batch_size 32 --kernel_v 1 --kernel_r 1 --num_return_sequences 1 --ckpt 1200 --timestamp 2022-07-20-21-53-50
