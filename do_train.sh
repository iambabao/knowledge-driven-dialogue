#!/usr/bin/env bash

python run.py \
  --model dual_ptrnet \
  --do_train \
  --epoch 10 \
  --batch 16 \
  --max_seq_length 256 \
  --max_triple_length 256 \
  --optimizer custom \
  --lr 0.01 \
  --dropout 0.3 \
  --pre_train_epochs 1 \
  --early_stop 10
