cd /data/szs/250010072/clash-for-linux
bash start.sh

source /etc/profile.d/clash.sh
proxy_on

cd /data/szs/250010072/gwy/Agent-Distillation/SDPO-main

export ANTHROPIC_BASE_URL=https://api.kimi.com/coding/
export ANTHROPIC_API_KEY=sk-kimi-sxMH7CdHtg2D8NbbzbFEpIh79lOhaPJfPFGhoddZZtFISjn8uxxbruwvuwONBi6T

#-i https://pypi.tuna.tsinghua.edu.cn/simple/
# https://github.com/verl-project/verl/blob/main/docker/verl0.6-cu128-torch2.8.0-fa2.7.4/Dockerfile.base
# https://github.com/Dao-AILab/flash-attention/releases
# https://github.com/vllm-project/vllm/releases?page=1
# https://github.com/lasgroup/SDPO
conda create -n sdpo2 python=3.10.0
conda activate vllm012exp
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 
pip install vllm-0.9.0+cu126-cp38-abi3-manylinux1_x86_64.whl
pip install "transformers<4.54.0"
pip install flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
python vllm_test.py

#
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install vllm-0.8.4+cu121-cp38-abi3-manylinux1_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl


#
/data/szs/250010072/gwy/Agent-Distillation/SDPO-main/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl


pip insatll 

# 设置环境变量以限制编译时的 CPU 核数（防止内存溢出），并强制从源码构建
MAX_JOBS=4 pip install flash-attn --no-cache-dir --no-build-isolation


#
conda create -n sdpo2 python=3.10.0
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install vllm-0.9.0+cu126-cp38-abi3-manylinux1_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

python -c "import flash_attn; print(flash_attn.__version__)"

python -c "import vllm; print(vllm.__version__)"
python -c "from vllm.v1.engine.utils import CoreEngineProcManager"



pip install flash_attn-2.7.1.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

#
pip install vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl
pip install flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl


source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate agent



#
conda create -n sdpo2 python=3.10.0
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install vllm-0.9.0+cu126-cp38-abi3-manylinux1_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl











