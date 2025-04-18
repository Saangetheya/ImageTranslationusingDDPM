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

Training with subject ids, need to specify:
- ```TOTAL_NUM_DIAGNOSIS_CLASS```: the total number of diagnosis classes (currently 3)
- ```VALIDATION_DIAGNOSIS_CLASS```: the diagnosis id class for validation (currently choice from 0, 1, 2)
```bash
accelerate launch train.py \
    --train_data_dir demo_data/train_data \
    --val_data_dir demo_data/eval_data \
    --output_dir YOUR_OUTPUT_DIR \
    --num_classes TOTAL_NUM_DIAGNOSIS_CLASS \
    --val_data_class VALIDATION_DIAGNOSIS_CLASS
```

### About 3D UNet / Spatial-Temporal UNet
Check the links below
- [3D UNet](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_3d_condition.py)
- [ST UNet](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_spatio_temporal_condition.py)

