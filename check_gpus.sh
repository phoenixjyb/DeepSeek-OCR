#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr
python -c "import torch; print('Number of GPUs:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
