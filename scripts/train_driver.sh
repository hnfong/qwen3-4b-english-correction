#!/bin/bash
RUN=${RUN:-r16a}
MODEL=${MODEL:-models/qwen3-4b-instruct}
BUDGET_MIN=${BUDGET_MIN:-22}
STEPS=${STEPS:-5}
END=$((SECONDS + BUDGET_MIN*60))
cd /Volumes/StudioData/projects/qwen-small-english-teacher
while [ $SECONDS -lt $END ]; do
  .venv/bin/python scripts/train_lora.py --run $RUN --max-examples 960 --batch 4 --accum 1 \
    --seqlen 160 --lr 1.5e-4 --max-minutes 6 --max-steps $STEPS --val-every 4 --save-every 100 \
    --model "$MODEL" --init-adapter out/lora_$RUN/last >> tmp/train_${RUN}_driver.log 2>&1
  echo "--- exited at $SECONDS s ---" >> tmp/train_${RUN}_driver.log
done
echo DRIVER_DONE >> tmp/train_${RUN}_driver.log
