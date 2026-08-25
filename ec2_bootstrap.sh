#!/usr/bin/env bash
# Set up a fresh EC2 GPU instance and run the same experiment as the local one.
#
#   ssh -i ~/.ssh/acm-key.pem ubuntu@<PUBLIC-IP>
#   curl -sL https://raw.githubusercontent.com/acmucsd-projects/su26-ai-team-1/merge-all/ec2_bootstrap.sh -o run.sh
#   BUCKET=s3://your-bucket bash run.sh
#
# Everything happens inside tmux, so the run survives a dropped SSH connection.
set -euo pipefail

BUCKET="${BUCKET:?set BUCKET, e.g. BUCKET=s3://acm-ai-team1-hmer}"
BRANCH="${BRANCH:-merge-all}"
EPOCHS="${EPOCHS:-4}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"

echo "==> environment"
source activate pytorch
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; \
           print('GPU:', torch.cuda.get_device_name(0))"

echo "==> code"
[ -d su26-ai-team-1 ] || git clone https://github.com/acmucsd-projects/su26-ai-team-1.git
cd su26-ai-team-1
git fetch origin && git checkout "$BRANCH" && git pull
pip install -q -r requirements.txt

echo "==> data"
if [ ! -d processed-augmented ]; then
  aws s3 cp "$BUCKET/data/processed-augmented.zip" .
  unzip -q processed-augmented.zip
fi

# The local experiment validates on CLEAN images while training on augmented
# ones -- augmented validation would not be comparable to the 63.9% baseline.
echo "==> assembling the same augmented-train / clean-valid view used locally"
if [ ! -d processed-mixed ]; then
  if [ -d processed ]; then
    mkdir -p processed-mixed/images processed-mixed/labels
    ln -sfn ../processed-augmented/vocab.json    processed-mixed/vocab.json
    ln -sfn ../processed-augmented/metadata.json processed-mixed/metadata.json
    for s in train synthetic symbols; do
      ln -sfn "../../processed-augmented/images/$s"        "processed-mixed/images/$s"
      ln -sfn "../../processed-augmented/labels/$s.jsonl"  "processed-mixed/labels/$s.jsonl"
    done
    ln -sfn ../../processed/images/valid       processed-mixed/images/valid
    ln -sfn ../../processed/labels/valid.jsonl processed-mixed/labels/valid.jsonl
  else
    echo "!! clean processed/ not uploaded -- falling back to augmented validation."
    echo "!! ExpRate will read ~4pp LOWER and is NOT comparable to the 63.9% baseline."
    ln -sfn processed-augmented processed-mixed
  fi
fi

echo "==> training (detach with Ctrl+B then D; reattach with: tmux attach -t train)"
tmux new -s train -d "
  python -u run_train.py \
    --processed processed-mixed \
    --train-split train,synthetic \
    --aux \
    --epochs $EPOCHS --batch-size $BATCH --device cuda --workers $WORKERS \
    --limit-val 1024 \
    --encoder-lr 3e-5 --decoder-lr 1e-4 \
    --checkpoint best_model_aws.pt 2>&1 | tee training_aws.log

  aws s3 cp best_model_aws.pt $BUCKET/checkpoints/
  aws s3 cp training_aws.log  $BUCKET/checkpoints/
  aws s3 cp history.json      $BUCKET/checkpoints/history_aws.json
  echo
  echo '=========================================================='
  echo ' DONE. Results are in $BUCKET/checkpoints/.'
  echo ' TERMINATE THE INSTANCE -- it bills ~\$29/day while idle.'
  echo '=========================================================='
"
sleep 5
tmux attach -t train
