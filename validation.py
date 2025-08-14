import os
from glob import glob
from typing import List
import cv2
import pandas as pd
from colorama import Fore, Style
from config import VALIDATION_LIMITS


def validate_file_exists(filepath: str, description: str) -> bool:
    """Check if a required file exists at the specified path.
    
    Args:
        filepath: Path to the file to check
        description: Human-readable description of the file for error messages
        
    Returns:
        bool: True if file exists, False otherwise
    """
    if not os.path.exists(filepath):
        print(f"{Fore.RED}{description} not found: {filepath}{Style.RESET_ALL}")
        return False
    return True


def validate_numeric_range(value: float, param_name: str, min_val: float, max_val: float) -> bool:
    """Validate that a numeric parameter is within the specified range.
    
    Args:
        value: The numeric value to validate
        param_name: Name of the parameter for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Returns:
        bool: True if value is within range, False otherwise
    """
    if not (min_val <= value <= max_val):
        print(
            f"{Fore.RED}{param_name} ({value}) outside valid range [{min_val}, {max_val}]{Style.RESET_ALL}"
        )
        return False
    return True


def validate_system_specs(specs_df: pd.DataFrame) -> bool:
    """Validate system specifications dataframe for required parameters and ranges.
    
    Performs comprehensive validation of all system specification parameters including
    geographical coordinates, calibration settings, time ranges, and system parameters.
    Ensures all values are within acceptable physical and operational limits.
    
    Args:
        specs_df (pd.DataFrame): DataFrame containing system specifications loaded
                               from the Excel file
                               
    Returns:
        bool: True if all specifications are valid, False if any validation fails
    """
    try:
        # Check latitude
        lat = float(specs_df["Lattitude (°)"][0])
        if not validate_numeric_range(lat, "Latitude", *VALIDATION_LIMITS["latitude"]):
            return False

        # Check longitude
        lon = float(specs_df["Longitude (°)"][0])
        if not validate_numeric_range(lon, "Longitude", *VALIDATION_LIMITS["longitude"]):
            return False

        # Check elevation
        elev = float(specs_df["Elevation (m)"][0])
        if not validate_numeric_range(elev, "Elevation", *VALIDATION_LIMITS["elevation"]):
            return False

        # Check positive values
        positive_params = [
            "Solar panel peak wattage (W)",
            "Batt nominal capacity (Ah)",
            "Batt nominal voltage (V)",
        ]
        for param in positive_params:
            if float(specs_df[param][0]) <= 0:
                print(f"{Fore.RED}{param} must be positive{Style.RESET_ALL}")
                return False

        # Check percentages
        percentage_params = ["Converter efficiency (%)", "Max SOC (%)", "Min SOC (%)"]
        for param in percentage_params:
            value = float(specs_df[param][0])
            if not validate_numeric_range(value, param, *VALIDATION_LIMITS["percentage_range"]):
                return False

        return True

    except (ValueError, KeyError, IndexError) as e:
        print(f"{Fore.RED}Error validating system specs: {e}{Style.RESET_ALL}")
        return False


def validate_images(image_dir: str, min_count: int = 1, extensions: List[str] = None) -> List[str]:
    """Validate and return list of valid image files in a directory.
    
    Scans the specified directory for image files with supported extensions,
    validates that they exist and are readable, and ensures minimum count requirements
    are met for processing stages like calibration.
    
    Args:
        image_dir (str): Path to the directory containing image files
        min_count (int, optional): Minimum number of images required. Defaults to 1.
        extensions (List[str], optional): List of valid file extensions. 
                                        Defaults to common image formats.
                                        
    Returns:
        List[str]: List of valid image file paths found in the directory.
                  Empty list if directory doesn't exist or insufficient images found.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
        print(f"{Fore.GREEN}Created directory: {image_dir}{Style.RESET_ALL}")

    images = []
    for ext in extensions:
        images.extend(glob(f"{image_dir}/*{ext}"))

    if len(images) < min_count:
        print(
            f"{Fore.RED}Need at least {min_count} images in {image_dir}, found {len(images)}{Style.RESET_ALL}"
        )
        return []

    return images


def get_user_choice(prompt: str, valid_choices: List[str]) -> str:
    """Get valid user input from a list of predefined choices.
    
    Continuously prompts the user until they provide a valid choice from the
    specified list. Provides clear feedback for invalid inputs and displays
    available options.
    
    Args:
        prompt (str): The message to display when asking for user input
        valid_choices (List[str]): List of acceptable input choices
        
    Returns:
        str: The user's validated choice from the valid_choices list
    """
    while True:
        choice = input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}").strip()
        if choice in valid_choices:
            return choice
        print(
            f"{Fore.RED}Invalid choice. Please enter one of: {', '.join(valid_choices)}{Style.RESET_ALL}"
        )


def prompt_for_file(message: str) -> None:
    """Prompt user to add required files and wait for confirmation.
    
    Displays a message asking the user to provide missing files and waits
    for them to press Enter when they've completed the requested action.
    
    Args:
        message (str): Instructions for what file(s) the user needs to provide
    """
    input(f"{Fore.CYAN}{message} Press Enter when ready...{Style.RESET_ALL}")


def validate_image_dimensions(sky_images: List[str], calib_images: List[str]) -> bool:
    """Validate that all images have consistent dimensions.
    
    Ensures all sky and calibration images have the same width and height,
    which is required for proper camera calibration and processing.
    
    Args:
        sky_images (List[str]): List of sky image file paths
        calib_images (List[str]): List of calibration image file paths
        
    Returns:
        bool: True if all images have matching dimensions, False otherwise
        
    Note:
        Returns True if no images are provided (empty lists).
    """
    all_images = sky_images + calib_images
    if not all_images:
        return True

    # Get reference dimensions from first image
    first_img = cv2.imread(all_images[0], cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print(f"{Fore.RED}Cannot read image: {os.path.basename(all_images[0])}{Style.RESET_ALL}")
        return False

    ref_height, ref_width = first_img.shape[:2]

    # Check all other images match reference dimensions
    for img_path in all_images[1:]:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{Fore.RED}Cannot read image: {os.path.basename(img_path)}{Style.RESET_ALL}")
            return False
        if img.shape[:2] != (ref_height, ref_width):
            print(
                f"{Fore.RED}Size mismatch: {os.path.basename(img_path)} ({img.shape[1]}x{img.shape[0]}) vs reference ({ref_width}x{ref_height}){Style.RESET_ALL}"
            )
            return False

    print(
        f"{Fore.GREEN}All {len(all_images)} images have consistent dimensions: {ref_width}x{ref_height}{Style.RESET_ALL}"
    )
    return True
