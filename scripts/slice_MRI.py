import nibabel as nib
import numpy as np
import cv2
import argparse
import os

DIM_DICT = {
    "sagittal": 1,
    "coronal": 0,
    "axial": 2
}

def load_data(path):
    scan = nib.load(path)
    volume = scan.get_fdata()
    min = np.amax(volume)
    max = np.amin(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def slice_data(volume, direction, slice_index):
    if direction == "sagittal":
        slice_image = volume[:, slice_index, :]
    elif direction == "coronal":
        slice_image = volume[slice_index, :, :]
    elif direction == "axial":
        slice_image = volume[:, :, slice_index]
    return slice_image
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1_path", type=str, required=True)
    parser.add_argument("--dwi_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "T1"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "DWI"), exist_ok=True)
    subject_id = args.t1_path.split("/")[-1].split("_")[0]

    t1_volume = load_data(args.t1_path)
    dwi_volume = load_data(args.dwi_path)

    with open(os.path.join(args.save_dir, "index.txt"), "w") as f:
        for direction in ["sagittal", "coronal", "axial"]:
            for idx in range(t1_volume.shape[DIM_DICT[direction]]):
                t1_slice = slice_data(t1_volume, direction, idx)
                if np.mean(t1_slice < 1.0) < 0.2:
                    continue
                dwi_slice = slice_data(dwi_volume, direction, idx)
                if np.mean(dwi_slice < 1.0) < 0.2:
                    continue

                cv2.imwrite(f"{args.save_dir}/T1/{subject_id}_{direction}_{idx}.png", t1_slice * 255)
                cv2.imwrite(f"{args.save_dir}/DWI/{subject_id}_{direction}_{idx}.png", dwi_slice * 255)

                f.write(f"{subject_id} T1/{subject_id}_{direction}_{idx}.png DWI/{subject_id}_{direction}_{idx}.png\n")

    f.close()
            
    

if __name__ == "__main__":
    main()