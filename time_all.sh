#!/bin/bash

# IMAGES=("1024x768.png" "1920x1200.png" "3840x2160.png" "720x480.png" "7680x4320.png")
# THREADS=(1 4 16 64)
# SEAMS=128
# RUNS=5
IMAGES=("1024x768.png" "1920x1200.png")
THREADS=(1 4)
SEAMS=128
RUNS=1

for img in "${IMAGES[@]}"; do
    for t_energy in "${THREADS[@]}"; do
        for t_cum in "${THREADS[@]}"; do
            for run in $(seq 1 $RUNS); do
                out_file="res/${img}_${t_energy}_${t_cum}_${run}.txt"
                ./run_prog.sh "$out_file" supafast.cpp "images/$img" "out_images/${img}_carved.jpg" "$SEAMS" "$t_energy" "$t_cum"
            done
        done
    done
done