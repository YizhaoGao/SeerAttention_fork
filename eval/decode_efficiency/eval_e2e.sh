#!/bin/bash

# Set the output folder (modify as needed)
OUT_DIR="./nsys_7b"
MODEL_DIR=Qwen/Qwen2.5-7B-Instruct
mkdir -p ${OUT_DIR}

# contexts=(8192 16384 32768 65536)
# batch_sizes=(16 8 4 2)

contexts=(128000)
batch_sizes=(1)

for idx in "${!contexts[@]}"; do
    context=${contexts[$idx]}
    bs=${batch_sizes[$idx]}
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas \
      -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
      --cudabacktrace=true -x true \
      -o "${OUT_DIR}/${context}_b${bs}_dense.nsys-rep" \
      python3 benchmark_e2e_single.py \
      --model_path ${MODEL_DIR} \
      --context_len ${context} \
      --decode_len 32 \
      --batch_size ${bs} \
      --dense_profile
done

for idx in "${!contexts[@]}"; do
    context=${contexts[$idx]}
    bs=${batch_sizes[$idx]}
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas \
      -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
      --cudabacktrace=true -x true \
      -o "${OUT_DIR}/${context}_b${bs}_sparse.nsys-rep" \
      python3 benchmark_e2e_single.py \
      --model_path ${MODEL_DIR} \
      --context_len ${context} \
      --decode_len 32 \
      --batch_size ${bs} 
done

# batch_sizes=(1 1 1 1)
batch_sizes=(1)
for idx in "${!contexts[@]}"; do
    context=${contexts[$idx]}
    bs=${batch_sizes[$idx]}
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas \
      -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
      --cudabacktrace=true -x true \
      -o "${OUT_DIR}/${context}_b${bs}_dense.nsys-rep" \
      python3 benchmark_e2e_single.py \
      --model_path ${MODEL_DIR} \
      --context_len ${context} \
      --decode_len 32 \
      --batch_size ${bs} \
      --dense_profile
done

for idx in "${!contexts[@]}"; do
    context=${contexts[$idx]}
    bs=${batch_sizes[$idx]}
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas \
      -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
      --cudabacktrace=true -x true \
      -o "${OUT_DIR}/${context}_b${bs}_sparse.nsys-rep" \
      python3 benchmark_e2e_single.py \
      --model_path ${MODEL_DIR} \
      --context_len ${context} \
      --decode_len 32 \
      --batch_size ${bs} 
done