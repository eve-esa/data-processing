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
pip install "paddlex[base]==3.1.4"
pip install paddlepaddle==3.2.2
pip install langchain==0.3.26
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
pip install -e .
pip install lmdeploy==0.9.2

cd MonkeyOCR
python tools/download_model.py -n MonkeyOCR-pro-3B

echo "Setup complete. MonkeyOCR environment ready."


# write a small helper to extract first pages of the pdfs so you dont have to run the whole thing on all pages
# after you setup, run the predictions using the following command: python3 parse.py <dir> --pred-abandon