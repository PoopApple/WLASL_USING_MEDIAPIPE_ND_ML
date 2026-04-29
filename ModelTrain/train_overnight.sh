#!/bin/bash
# Overnight training — 500 words then 750 words
# Run with: bash train_overnight.sh
# For TFRecord mode: bash train_overnight.sh --tfrecord

set -e

USE_TFRECORD=""
if [ "$1" = "--tfrecord" ]; then
    USE_TFRECORD="--use_tfrecord"
    echo ">>> TFRecord mode enabled"
    echo ">>> Converting datasets to TFRecords first..."
    .venv/bin/python convert_to_tfrecord.py --num_words 500
    .venv/bin/python convert_to_tfrecord.py --num_words 750
fi

echo "=========================================="
echo "Starting overnight training"
echo "Time: $(date)"
echo "=========================================="

echo ""
echo ">>> Run 1: 500 words"
echo "=========================================="
.venv/bin/python train_v2.py \
    --num_words 500 \
    --batch_size 128 \
    --epochs 60 \
    --patience 15 \
    --lr 1e-3 \
    $USE_TFRECORD

echo ""
echo ">>> Run 1 complete at $(date)"
echo "=========================================="

echo ""
echo ">>> Run 2: 750 words"
echo "=========================================="
.venv/bin/python train_v2.py \
    --num_words 750 \
    --batch_size 128 \
    --epochs 60 \
    --patience 15 \
    --lr 1e-3 \
    $USE_TFRECORD

echo ""
echo ">>> Run 2 complete at $(date)"
echo "=========================================="
echo "All training complete at $(date)"
