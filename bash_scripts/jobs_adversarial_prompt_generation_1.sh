#!/bin/sh

for num_adv in 0 4 8 12 16 # 0 2 4 6 8 10 12 14 16 18 20
do
    echo "num_adv $num_adv:"
    # batch size needs to be slightly lower than 350, cuda memory problems at 114th prompt in 20 tokens long adversarial prompt
    python gcg.py -d $1 -b 200 --model_wt_path $2 --num_iters 100 --num_adv $num_adv 
done