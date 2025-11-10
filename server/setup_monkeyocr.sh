#!/bin/bash

if [ ! -d "MonkeyOCR" ]; then
  echo "Cloning MonkeyOCR repository..."
  git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
else
  echo "MonkeyOCR directory already exists. Skipping clone."
fi

cd MonkeyOCR

# set your CUDA version here
CUDA_VERSION=126

echo "Using CUDA version: $CUDA_VERSION"


pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu${CUDA_VERSION}/
pip install langchain==0.3.26
pip install "paddlex[base]==3.1.4"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
pip install -e .
pip install lmdeploy==0.9.2

echo "Setup complete. MonkeyOCR environment ready."
