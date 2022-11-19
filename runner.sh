#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
export TVM_NUM_THREADS=8
module load cuda/10.2.0

conda activate gpu
cd /jet/home/jshah2/HW4/NNI-distillation

python main.py

mkdir models
mv distilled_model.onnx models/

echo "PERFORMING FIRST COMPILE"
tvmc compile --target "llvm" --input-shapes "data:[64,1,28,28]" --output models/tvm_first_compile.tar models/distilled_model.onnx

tar -xvf models/tvm_first_compile.tar -C models/tvm_first_compile

echo "TUNING MODEL"
tvmc tune --target "llvm" --output models/tune_config.json models/distilled_model.onnx

echo "COMPILING TUNED MODEL"

tvmc compile --target "llvm" --input-shapes "data:[64,1,28,28]" --tuning-records models/tune_config.json --output models/final_tuned_model.tar models/distilled_model.onnx

python get_image.py

tvmc run --inputs imagenet_cat.npz --print-time --output temp_predictions.npz models/final_tuned_model
