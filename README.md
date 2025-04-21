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
accelerate launch train_3d.py \
    --train_data_dir demo_data/train_data \
    --val_data_dir demo_data/eval_data \
    --output_dir YOUR_OUTPUT_DIR
```
### Start inference
```bash
python inference.py \
    --model_path YOUR_OUTPUT_DIR \
    --t1_dir demo_data/eval_data \
    --dwi_dir demo_data/eval_data \
    --output_dir inference_results \
    --temporal_length 3

