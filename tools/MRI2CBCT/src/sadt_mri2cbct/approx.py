#!/usr/bin/env python-real

import os
import shutil

import logging

logger = logging.getLogger(__name__)

from .approximate import approximation
from .crop_approximation import crop_volume, get_transformation

def create_folder(folder):
    """
    Creates a folder if it does not already exist.

    Arguments:
    folder (str): Path of the folder to create.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def run_script_first_approximation(cbct_folder, mri_folder, output_folder, model_folder, tmp_folder):
    """
    Approximates CBCT images to MRI images and saves the resulting images.

    Args:
        cbct_folder (str): Path to the folder containing CBCT images
        mri_folder (str): Path to the folder containing MRI images
        output_folder (str): Path to the folder where output images will be saved
        model_folder (str): Path to the nnUNet condyle segmentation model
        tmp_folder (str): Scratch folder for nnUNet per-case files

    Returns:
        str: Path to the folder containing the approximated images.
    """

    first_approximation_folder = os.path.join(output_folder, "first_approximation")
    create_folder(first_approximation_folder)

    approximation(cbct_folder, mri_folder, first_approximation_folder, model_folder, tmp_folder)
    return first_approximation_folder

def run_script_get_transformation(mean_folder, cbct_folder, output_folder):
    """
    Generates the registration of the CBCT images to the mean CBCT and saves the results.

    Args:
        mean_folder (str): Path to the folder containing the mean CBCT image.
        cbct_folder (str): Path to the folder containing the CBCT images.
        output_folder (str): Path to the folder where the registered images will be saved.

    Returns:
        str: Path to the folder containing the registered images.
    """
    
    transformation_folder = os.path.join(output_folder, "mean_registration")
    create_folder(transformation_folder)
    get_transformation(mean_folder, cbct_folder, transformation_folder)
    return transformation_folder

def delete_folder(folder_path):
    """
    Deletes a folder if it exists.

    Arguments:
    folder_path (str): Path of the folder to create.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        logger.info(f"The folder '{folder_path}' has been deleted successfully.")
    else:
        logger.info(f"The folder '{folder_path}' does not exist.")
        
def run_script_crop_volumes(ROI_file, transformation_folder, first_approximation_folder, cbct_folder, output_folder):
    """
    Crops the CBCT volumes and MRI volumes based on the ROI and saves the results.

    Args:
        ROI_file (str): Path to the file containing the ROI.
        transformation_folder (str): Path to the folder containing the transformation files.
        first_approximation_folder (str): Path to the folder containing the approximated images.
        mri_folder (str): Path to the folder containing the MRI images.
        output_folder (str): Path to the folder where the cropped images will be saved.

    Returns:
        str: Path to the folder containing the cropped images.
    """
    
    cropped_cbct_folder = os.path.join(output_folder, "cropped_cbct")
    create_folder(cropped_cbct_folder)
    crop_volume(ROI_file, transformation_folder, first_approximation_folder, cbct_folder, cropped_cbct_folder)

