#!/usr/bin/env python-real

import os
import SimpleITK as sitk

import logging

logger = logging.getLogger(__name__)

def extract_id(filename):
    """
    Extracts and returns the ID from a filename, removing common NIfTI extensions.
    
    Parameters:
        filename (str): The filename from which to extract the ID.
    
    Returns:
        str: The extracted ID without the extension.
    """
    # Remove the extension using os.path.splitext
    type_file = 0
    base = os.path.splitext(filename)[0]
    # If the file has a double extension (commonly .nii.gz), remove the second extension
    if base.endswith('.nii'):
        base = os.path.splitext(base)[0]
        type_file=1
    
    
    return base,type_file

def calculate_new_origin(image):
    """
    Calculate the new origin to center the image in the Slicer viewport across all axes.
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    # Calculate the center offset for each axis
    new_origin = [(size[i] * spacing[i]) / 2 for i in range(len(size))]
    new_origin = [new_origin[2],-new_origin[0],new_origin[1]] # FOR MRI
    return tuple(new_origin)

def modify_image_properties(nifti_file_path, new_direction, output_file_path=None, acquisition_z_spacing=3.0):
    """
    Read a NIfTI file, change its Direction and optionally center and save the modified image.
    """
    image = sitk.ReadImage(nifti_file_path)
    # Set the new direction
    image.SetDirection(new_direction)
    spacing = list(image.GetSpacing())
    
    # Update only Z spacing (index 2)
    if acquisition_z_spacing != "None":
        spacing[2] = float(acquisition_z_spacing)
        image.SetSpacing(tuple(spacing))

    # Calculate and set the new origin
    new_origin = calculate_new_origin(image)
    image.SetOrigin(new_origin)

    if output_file_path:
        sitk.WriteImage(image, output_file_path)
        logger.info(f"Modified image saved to {output_file_path}")

    return image

def orient(input_folder, direction, output_folder, acquisition_z_spacing="None"):
    """Set every MRI's direction, Z spacing and origin, writing `<name>_OR`.

    Upstream's `main(args)`, with the argparse namespace unpacked into
    parameters and its per-file 0.5 s sleep gone -- that sleep existed to let
    Slicer's progress bar redraw, and costs twenty seconds on a cohort of forty.
    """
    new_direction = tuple(map(float, str(direction).split(',')))  # comma-separated values
    output_folder = output_folder if output_folder else input_folder

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))

    # Process each file
    for file_path in nifti_files:
        filename = os.path.basename(file_path)
        file_id,type_file = extract_id(filename)
        if type_file==0:
            output_file_path = os.path.join(output_folder, f"{file_id}_OR.nii")
        else :
            output_file_path = os.path.join(output_folder, f"{file_id}_OR.nii.gz")
        modify_image_properties(file_path, new_direction, output_file_path, acquisition_z_spacing)

