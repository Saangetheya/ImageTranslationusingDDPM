# ImageTranslationusingDDPM

### Environment
```bash
pip install -r requirements.txt
```

### Setup GPUs and machines
```bash
accelerate config
```

### Start training
```bash
accelerate launch train.py \
    --train_data_dir demo_data/train_data \
    --val_data_dir demo_data/eval_data \
    --output_dir YOUR_OUTPUT_DIR
```