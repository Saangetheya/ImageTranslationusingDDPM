import torch
import cv2
import os
import numpy as np
import argparse

from pipeline_3d import MRIs3DDDPMPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="YOUR_OUTPUT_DIR",
                    help="Path to the trained model")
parser.add_argument("--t1_dir", type=str, default="demo_data/eval_data",
                    help="Directory containing T1 images")
parser.add_argument("--dwi_dir", type=str, default="demo_data/eval_data",
                    help="Directory containing DWI images")
parser.add_argument("--output_dir", type=str, default="inference_results",
                    help="Directory to save output images")
parser.add_argument("--temporal_length", type=int, default=3,
                    help="Number of consecutive slices to use")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Get and sort image file list
t1_image_list = sorted([f for f in os.listdir(args.t1_dir) if f.endswith('.png')])
print(f"Found {len(t1_image_list)} images in T1 directory: {t1_image_list}")

# Group by direction
direction_groups = {}
for img_name in t1_image_list:
    # Assume format: subject_id_direction_idx.png
    parts = img_name.split('_')
    if len(parts) < 3:
        print(f"Warning: image {img_name} does not follow the expected naming format")
        continue

    subject_id = parts[0]
    direction = parts[1]
    try:
        slice_idx = int(parts[2].split('.')[0])
    except ValueError:
        print(f"Warning: could not parse slice index from {img_name}")
        continue

    key = f"{subject_id}_{direction}"
    if key not in direction_groups:
        direction_groups[key] = []

    direction_groups[key].append({
        'filename': img_name,
        'slice_idx': slice_idx
    })

# Print debug info about direction groups
print(f"Found direction groups: {list(direction_groups.keys())}")
for key, slices in direction_groups.items():
    print(f"Group {key} has {len(slices)} slices")

# Sort slices within each group by index
for key in direction_groups:
    direction_groups[key] = sorted(direction_groups[key], key=lambda x: x['slice_idx'])

# Create consecutive slice sequences
valid_sequences = []
for key, slices in direction_groups.items():
    if len(slices) >= args.temporal_length:
        for i in range(len(slices) - args.temporal_length + 1):
            valid_sequences.append((key, i, [s['filename'] for s in slices[i:i + args.temporal_length]]))

print(f"Valid sequences found: {len(valid_sequences)}")
for i, (key, start_idx, filenames) in enumerate(valid_sequences):
    print(f"Sequence {i}: key={key}, start_idx={start_idx}, files={filenames}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using CUDA for inference")
else:
    print("Using CPU for inference")

pipeline = MRIs3DDDPMPipeline.from_pretrained(args.model_path).to(device)

with open(os.path.join(args.output_dir, "result.txt"), "w") as f:
    for seq_idx, (key, start_idx, filenames) in enumerate(valid_sequences):
        print(f"Processing sequence {seq_idx}: {key} starting at {start_idx}")
        # Load consecutive slices
        t1_sequence = []
        dwi_sequence = []  # For comparison

        for filename in filenames:
            print(f"  Loading file: {filename}")
            # Load T1 image
            t1_path = os.path.join(args.t1_dir, filename)
            if not os.path.exists(t1_path):
                print(f"  Warning: T1 file not found: {t1_path}")
                continue

            t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
            h, w = t1_img.shape
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2
            t1_img = t1_img[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]
            t1_img_np = cv2.resize(t1_img, (64, 64), interpolation=cv2.INTER_LINEAR)
            t1_tensor = 1 - torch.from_numpy(t1_img_np).float() / 255.0
            t1_tensor = t1_tensor.unsqueeze(0) * 2 - 1
            t1_sequence.append(t1_tensor)

            # Try to load corresponding DWI image (for comparison)
            dwi_filename = filename.replace("T1", "DWI") if "T1" in filename else filename
            dwi_path = os.path.join(args.dwi_dir, dwi_filename)
            if os.path.exists(dwi_path):
                dwi_img = cv2.imread(dwi_path, cv2.IMREAD_GRAYSCALE)
                dwi_img = dwi_img[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]
                dwi_img = cv2.resize(dwi_img, (64, 64), interpolation=cv2.INTER_LINEAR)
                dwi_sequence.append(dwi_img)
            else:
                print(f"  Warning: DWI file not found: {dwi_path}")

        if len(t1_sequence) == 0:
            print(f"  Error: No T1 images loaded for sequence {seq_idx}")
            continue

        if len(t1_sequence) < args.temporal_length:
            print(f"  Warning: Only loaded {len(t1_sequence)} images, less than temporal_length {args.temporal_length}")

        # Create 3D input [1, C, T, H, W]
        t1_volume = torch.stack(t1_sequence, dim=1).unsqueeze(0)

        # Add debugging output
        print(f"  Input volume shape: {t1_volume.shape}")

        # Generate DWI images
        with torch.no_grad():
            generated_volumes = pipeline(
                t1_volume.to(device),
                batch_size=1,
                num_inference_steps=200,  # Same as training
                output_type="np"
            ).images

        # Add debugging output
        print(f"  Generated volumes shape: {generated_volumes.shape}")

        # Determine the format of the generated volumes
        if len(generated_volumes.shape) == 5:  # 5D tensor
            if generated_volumes.shape[1] > generated_volumes.shape[2]:  # [B, T, C, H, W]
                print("  Detected [B, T, C, H, W] format")
                temporal_dim = 1
                channel_dim = 2
            else:  # [B, C, T, H, W]
                print("  Detected [B, C, T, H, W] format")
                temporal_dim = 2
                channel_dim = 1
        else:
            print(f"  Unexpected shape: {generated_volumes.shape}")
            continue

        # Get temporal dimension size
        temporal_size = generated_volumes.shape[temporal_dim]
        print(f"  Temporal dimension size: {temporal_size}")

        # Save results
        subject_id, direction = key.split('_')

        # Process each slice in the temporal dimension
        for i in range(min(args.temporal_length, temporal_size)):
            print(f"  Processing slice {i}")

            # Get generated slice based on detected format
            if temporal_dim == 1:  # [B, T, C, H, W]
                if generated_volumes.shape[channel_dim] == 1:
                    gen_img = ((1 - generated_volumes[0, i, 0]) * 255).astype(np.uint8)
                else:
                    gen_img = ((1 - generated_volumes[0, i]) * 255).astype(np.uint8)
            else:  # [B, C, T, H, W]
                gen_img = ((1 - generated_volumes[0, 0, i]) * 255).astype(np.uint8)

            # Create comparison image
            t1_ori = t1_sequence[i].squeeze().cpu().numpy()
            t1_ori = ((1 - t1_ori) * 255 / 2 + 127.5).astype(np.uint8)

            if i < len(dwi_sequence):
                dwi_ori = dwi_sequence[i]
                comparison = np.concatenate([t1_ori, dwi_ori, gen_img], axis=1)
            else:
                comparison = np.concatenate([t1_ori, gen_img], axis=1)

            # Save image
            output_filename = f"{subject_id}_{direction}_seq{seq_idx}_slice{i}.png"
            cv2.imwrite(os.path.join(args.output_dir, output_filename), comparison)
            print(f"  Saved image: {output_filename}")

            f.write(f"Generated: {output_filename}\n")

print("Inference completed!")