#!/usr/bin/env python-real

import os
import glob

import logging

logger = logging.getLogger(__name__)


from .LR_crop import crop_cbct, crop_mri

            
def crop_folder(input_folder,output_folder, is_cbct=False):
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all .nii and .nii.gz files
    files = glob.glob(os.path.join(input_folder, "*.nii")) + glob.glob(os.path.join(input_folder, "*.nii.gz"))
    total_patients = len(files)
    patient_count = 0

    logger.info(f"[INFO] Found {total_patients} file(s) in {input_folder}.")

    for img_path in files:
        try:
            if is_cbct:
                crop_cbct(img_path, output_folder)
            else:
                crop_mri(img_path, output_folder)

            patient_count += 1

        except Exception as e:
            logger.error(f"[ERROR] Failed to process {img_path}: {e}")


