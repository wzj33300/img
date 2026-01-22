#!/bin/bash

KATAGO_BACKEND=$1
WEIGHT_FILE=$2
USE_HIGHTHREADS=$3
RELEASE_VERSION=2.0.0
GPU_NAME=`nvidia-smi -q | grep "Product Name" | cut -d":" -f2 | tr -cd '[:alnum:]._-'`
#GPU_NAME=TeslaT4

detect_auto_backend () {
  if [ "$GPU_NAME" == "TeslaT4" ]
  then
    KATAGO_BACKEND="CUDA"
  else
    KATAGO_BACKEND="OPENCL"
  fi
}

detect_auto_weight () {
  if [ "$GPU_NAME" == "TeslaK80" ]
  then
    WEIGHT_FILE="20b"
  elif [ "$GPU_NAME" == "TeslaP4" ]
  then
    WEIGHT_FILE="20b"
  else
    WEIGHT_FILE="18b"
  fi
}

if [ "$KATAGO_BACKEND" == "AUTO" ]
then
  detect_auto_backend
fi

if [ "$WEIGHT_FILE" == "AUTO" ]
then
  detect_auto_weight
fi

BLOCK=`echo $WEIGHT_FILE | sed -n "s/^.*b\([0-9]\+\).*$/\1/p"`
if [ "$BLOCK" == "" ]
then
  BLOCK=`echo $WEIGHT_FILE | sed -n "s/^[^0-9]*\([0-9]\+\)b.*$/\1/p"`
fi
if [ "$BLOCK" == "" ]
then
  BLOCK="18"
fi
if [ "$KATAGO_BACKEND" == "TRT" ]
then
  KATAGO_BACKEND="TENSORRT"
fi
echo "Using GPU: " $GPU_NAME
echo "Using Katago Backend: " $KATAGO_BACKEND
echo "Using Katago Weight: " $WEIGHT_FILE

cd /content
apt install --yes libzip4 1>/dev/null
rm -rf work
wget --quiet https://github.com/kinfkong/ikatago-for-colab/releases/download/$RELEASE_VERSION/work.zip -O work.zip
unzip -qq work.zip

cd /content/work
mkdir -p /content/work/data/bins
mkdir -p /content/work/data/weights
#download the binarires
if [ "$KATAGO_BACKEND" == "TENSORRT" ]
then
wget -O "katago.zip" "https://github.com/lightvector/KataGo/releases/download/v1.16.4/katago-v1.16.4-trt10.2.0-cuda12.5-linux-x64.zip"
unzip -o "katago.zip" -d "./katago_tmp"
cp ./katago_tmp/katago ./data/bins/katago
elif [ "$KATAGO_BACKEND" == "CUDA" ]
then
wget -O "katago.zip" "https://github.com/lightvector/KataGo/releases/download/v1.16.4/katago-v1.16.4-cuda12.8-cudnn9.8.0-linux-x64.zip"
unzip -o "katago.zip" -d "./katago_tmp"
cp ./katago_tmp/katago ./data/bins/katago
else
wget https://github.com/kinfkong/ikatago-for-colab/releases/download/$RELEASE_VERSION/katago-$KATAGO_BACKEND -O ./data/bins/katago
fi

chmod +x ./data/bins/katago
mkdir -p /root/.katago/
cp -r ./opencltuning /root/.katago/

#download the weights
if [ "$BLOCK" == "20" ]
then
  wget --quiet https://github.com/lightvector/KataGo/releases/download/v1.4.5/g170e-b20c256x2-s5303129600-d1228401921.bin.gz -O "./data/weights/"$BLOCK"b.bin.gz"
elif [ "$BLOCK" == "30" ]
then
  wget --quiet https://github.com/lightvector/KataGo/releases/download/v1.4.5/g170-b30c320x2-s4824661760-d1229536699.bin.gz -O "./data/weights/"$BLOCK"b.bin.gz"
else
  wget --quiet https://github.com/kinfkong/ikatago-for-colab/releases/download/$RELEASE_VERSION/kata1_weights.py -O kata1_weights.py
  python kata1_weights.py $WEIGHT_FILE
fi
cp /root/.katago/opencltuning/tune6_gpuTeslaK80_x19_y19_c256_mv8.txt /root/.katago/opencltuning/tune6_gpuTeslaK80_x19_y19_c256_mv10.txt
cp /root/.katago/opencltuning/tune6_gpuTeslaP100PCIE16GB_x19_y19_c256_mv8.txt /root/.katago/opencltuning/tune6_gpuTeslaP100PCIE16GB_x19_y19_c256_mv10.txt
cp /root/.katago/opencltuning/tune6_gpuTeslaP100PCIE16GB_x19_y19_c384_mv8.txt /root/.katago/opencltuning/tune6_gpuTeslaP100PCIE16GB_x19_y19_c384_mv10.txt
cp /root/.katago/opencltuning/tune8_gpuTeslaK80_x19_y19_c256_mv8.txt /root/.katago/opencltuning/tune8_gpuTeslaK80_x19_y19_c256_mv10.txt
cp /root/.katago/opencltuning/tune8_gpuTeslaP100PCIE16GB_x19_y19_c256_mv8.txt /root/.katago/opencltuning/tune8_gpuTeslaP100PCIE16GB_x19_y19_c256_mv10.txt

if [ "$KATAGO_BACKEND" == "TENSORRT" ]
then
  #  apt-get install libnvinfer8=8.2.0-1+cuda11.4
  apt-get install libnvinfer10=10.2.0.19-1+cuda12.5 libnvonnxparsers10=10.2.0.19-1+cuda12.5 libnvinfer-plugin10=10.2.0.19-1+cuda12.5
  wget https://github.com/kinfkong/ikatago-for-colab/releases/download/$RELEASE_VERSION/libssl1.1_1.1.1f-1ubuntu2.17_amd64.deb
  dpkg -i libssl1.1_1.1.1f-1ubuntu2.17_amd64.deb
fi
chmod +x ./change-config.sh

./change-config.sh $BLOCK"b" "./data/weights/"$BLOCK"b.bin.gz"

chmod +x ./ikatago-server