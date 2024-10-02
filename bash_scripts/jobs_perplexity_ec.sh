#!/bin/sh

for num_iters in 0 3 6 9 12
do
    for num_adv in 0 4 8 12 16 20
    do
        python perplexity_ec.py \
            --device $1 \
            --model_path $2 \
            --prompts_file data/adv_100it_350b/adversarial_prompts_t_${num_adv}.txt \
            --num_iters $num_iters \
            --num_adv $num_adv \
            --results_file results/perplexity_ec_distilled_results.json
    done
done