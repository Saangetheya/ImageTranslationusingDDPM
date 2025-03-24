# Pre-process Data

Make sure the raw T1 and DWI files ```.nii``` paired files are located in the same folder. For example:

```bash
YOUR_DATA_ROOT
---- 041_S_4143_20180712_A3_DWI_S55_MDtoENIGMA.nii
---- 041_S_4143_20180712_A3_T1_3T_accel_Preproc_fslreor_bet_biascorr_9DOF_aligned_2mm.nii
---- 130_S_4352_20190611_A3_DWI_P33_MDtoENIGMA.nii
---- 130_S_4352_20190611_A3_T1_3T_accel_Preproc_fslreor_bet_biascorr_9DOF_aligned_2mm.nii
```
### Run the script
```bash
python slice_MRI.py \
    --data_dir $YOUR_DATA_ROOT \
    --save_dir $YOUR_SAVE_DIR
```
After running the script, you should have the following under ```YOUR_SAVE_DIR```: ```T1``` folder contains all the T1 images, ```DWI``` folder contains all the DWI images and ```index.txt``` file includes the dataset index file which will be read by the dataloader.
```bash
YOUR_SAVE_DIR
---- T1\
---- DWI\
---- index.txt
```