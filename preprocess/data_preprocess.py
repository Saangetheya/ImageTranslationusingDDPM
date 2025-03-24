import nibabel as nib
import numpy as np
import cv2

path = "/wekafs/ict/wenbinte/data/MRIs/041_S_4143_20180712_A3_DWI_S55_MDtoENIGMA.nii"
scan = nib.load(path)
volume = scan.get_fdata()
min = np.amax(volume)
max = np.amin(volume)
volume = (volume - min) / (max - min)
volume = volume.astype("float32")

# print(volume.shape, volume.min(), volume.max(), volume.mean())
slice_image = volume[15, :, :]
cv2.imwrite("slice_image_x.png", slice_image * 255)