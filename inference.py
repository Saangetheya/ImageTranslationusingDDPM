import torch
import cv2
import os
import numpy as np
import argparse
import piq

from pipeline import MRIsDDPMPipeline


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/wekafs/ict/wenbinte/data/MRIs/output",
                    help="Path to the trained model")
parser.add_argument("--t1_dir", type=str, default="/wekafs/ict/wenbinte/data/MRIs/dataset/T1",
                    help="Directory containing T1 images")
parser.add_argument("--dwi_dir", type=str, default="/wekafs/ict/wenbinte/data/MRIs/dataset/DWI", 
                    help="Directory containing DWI images")
parser.add_argument("--output_dir", type=str, default="/wekafs/ict/wenbinte/data/MRIs/output/test",
                    help="Directory to save output images")
args = parser.parse_args()

model_path = args.model_path
t1_image_list = sorted(os.listdir(args.t1_dir))
dwi_image_list = sorted(os.listdir(args.dwi_dir))
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

t1_images = [cv2.imread(os.path.join(args.t1_dir, t1_image_list[i]), cv2.IMREAD_GRAYSCALE) for i in range(len(t1_image_list))]
h, w = t1_images[0].shape
        
# Calculate dimensions for center crop
min_dim = min(h, w)
start_h = (h - min_dim) // 2
start_w = (w - min_dim) // 2

# Perform center crop
t1_images = [t1_image[start_h:(start_h + min_dim), start_w:(start_w + min_dim)] for t1_image in t1_images]
t1_images_ori_np = [cv2.resize(t1_image, (64, 64), interpolation=cv2.INTER_LINEAR) for t1_image in t1_images]
t1_images = [(1 - torch.from_numpy(t1_image).float().unsqueeze(0).unsqueeze(0) / 255.) * 2 - 1. for t1_image in t1_images_ori_np]

dwi_images = [cv2.imread(os.path.join(args.dwi_dir, dwi_image_list[i]), cv2.IMREAD_GRAYSCALE) for i in range(len(dwi_image_list))]
dwi_images = [dwi_image[start_h:(start_h + min_dim), start_w:(start_w + min_dim)] for dwi_image in dwi_images]
dwi_images_ori_np = [cv2.resize(dwi_image, (64, 64), interpolation=cv2.INTER_LINEAR) for dwi_image in dwi_images]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = MRIsDDPMPipeline.from_pretrained(model_path).to(device)
fid_metric = piq.FID().to(device)

with open(os.path.join(output_dir, "result.txt"), "w") as f:
    for i, t1_image in enumerate(t1_images):
        image = pipeline(t1_image.to(device), batch_size=1, num_inference_steps=200, output_type="np").images
        image_processed = ((1-image) * 255).round().astype("uint8")[0, ..., 0]

        image_combine = np.concatenate([t1_images_ori_np[i], dwi_images_ori_np[i], image_processed], axis=1)

        gen_image = torch.tensor(image_processed).unsqueeze(0).unsqueeze(0)
        ref_image = torch.tensor(t1_images_ori_np[i]).unsqueeze(0).unsqueeze(0)

        # ssim_value = piq.ssim(gen_image, ref_image, data_range=1.0)
        # print(f"SSIM: {ssim_value}")

        # psnr_value = piq.psnr(gen_image, ref_image, data_range=1.0)
        # print(f"PSNR: {psnr_value}")

        # lpips_value = piq.LPIPS(replace_pooling=True)(gen_image, ref_image)
        # print(f'LPIPS: {lpips_value.item()}')

        # l1_loss = torch.nn.functional.l1_loss(gen_image, ref_image)
        # print(f'L1 Loss: {l1_loss.item()}')

        # fid_value = fid_metric(gen_image, ref_image)
        # print(f'FID: {fid_value.item()}')

        # f.write(f"{i:03d}: SSIM: {ssim_value}, PSNR: {psnr_value}, LPIPS: {lpips_value.item()}, L1 Loss: {l1_loss.item()}, FID: {fid_value.item()}\n")

        cv2.imwrite(os.path.join(output_dir, f"{i:03d}.png"), image_combine)











