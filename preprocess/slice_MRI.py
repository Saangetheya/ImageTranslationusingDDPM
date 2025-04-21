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

def get_paired_data_path_from_subject_id(data_list, subject_id):
    paths = [path for path in data_list if path.split("_")[0] == subject_id]
    assert len(paths) == 2
    # sort as T1, DWI
    t1_path = [path for path in paths if "T1" in path][0]
    dwi_path = [path for path in paths if "DWI" in path][0]
    return t1_path, dwi_path

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
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "T1"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "DWI"), exist_ok=True)

    data_list = os.listdir(args.data_dir)
    subject_ids = list(set([data.split("_")[0] for data in data_list]))
    
    with open(os.path.join(args.save_dir, "index.txt"), "w") as f:
        for subject_id in subject_ids:
            t1_path, dwi_path = get_paired_data_path_from_subject_id(data_list, subject_id)
            t1_volume = load_data(os.path.join(args.data_dir, t1_path))
            dwi_volume = load_data(os.path.join(args.data_dir, dwi_path))

            for direction in ["sagittal", "coronal", "axial"]:
                for idx in range(t1_volume.shape[DIM_DICT[direction]]):
                    t1_slice = slice_data(t1_volume, direction, idx)
                    if np.mean(t1_slice < 1.0) < 0.05:
                        continue
                    dwi_slice = slice_data(dwi_volume, direction, idx)
                    if np.mean(dwi_slice < 1.0) < 0.05:
                        continue

                    cv2.imwrite(f"{args.save_dir}/T1/{subject_id}_{direction}_{idx}.png", t1_slice * 255)
                    cv2.imwrite(f"{args.save_dir}/DWI/{subject_id}_{direction}_{idx}.png", dwi_slice * 255)

                    f.write(f"{subject_id} T1/{subject_id}_{direction}_{idx}.png DWI/{subject_id}_{direction}_{idx}.png\n")

    f.close()
            
    

if __name__ == "__main__":
    main()