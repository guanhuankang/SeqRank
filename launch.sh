#!/bin/bash
pip install flask
pip install werkzeug

if python -c "import torch; print(torch.cuda.is_available())"; then
    device="cuda"
else
    device="cpu"
fi

model_weight=$1

config_file="configs/swinL.yaml"
if echo $model_weight | grep "correct_ckp.pth"; then
    config_file="configs/swinL_correct.yaml"
fi

echo $model_weight, $config_file
python -m online.app --config-file $config_file --output output --opts MODEL.WEIGHTS $model_weight MODEL.DEVICE $device

