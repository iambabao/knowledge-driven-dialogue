#!/usr/bin/env bash

python run.py \
  --model dual_ptrnet \
  --do_test \
  --batch 32 \
  --max_seq_length 256 \
  --max_triple_length 256
