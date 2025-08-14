from typing import List, Tuple, Optional
import cv2
import os
from glob import glob
import numpy as np
from numpy import tan
import yaml
from scipy.optimize import minimize
from colorama import Fore, Style
from config import PATHS
from camera_coords_to_image_intrinsic import camera_coords_to_image_intrinsic


def create_image(
    fov_angle: float,
    poly_incident_angle_to_radius: List[float],
    principal_point: List[float]
) -> None:
    """
    Create and save a visualization of the FOV on a sample image.

    Draws FOV circle and angle markers on the first available image to help
    validate the FOV estimation.

    Args:
        fov_angle: Estimated field of view angle in degrees
        poly_incident_angle_to_radius: Polynomial coefficients for angle to radius mapping
        principal_point: Image center coordinates [x, y]
    """
    # Find available images
    folder = PATHS['sky_images']
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [
        f for f in glob(os.path.join(folder, "*"))
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not images:
        print(f"{Fore.RED}No images found in {folder}{Style.RESET_ALL}")
        return

    # Load first image and calculate FOV circle parameters
    image = cv2.imread(images[0])
    if image is None:
        print(f"{Fore.RED}Failed to load image: {images[0]}{Style.RESET_ALL}")
        return

    # Calculate FOV circle radius
    theta = np.deg2rad([fov_angle])
    x_prime = tan(theta)
    fov_limit = camera_coords_to_image_intrinsic(
        np.column_stack((x_prime, [0])),
        poly_incident_angle_to_radius,
        principal_point
    )

    distance_to_fov = fov_limit[0][0] - principal_point[0]
    center_point = (round(principal_point[0]), round(principal_point[1]))

    # Draw FOV circle
    image = cv2.circle(
        image,
        center_point,
        round(distance_to_fov),
        (0, 0, 255),  # Red color in BGR
        2  # Line thickness
    )

    def add_fov_text(img: np.ndarray, text: str, position: Tuple[int, int]) -> np.ndarray:
        """Helper function to add FOV text consistently"""
        return cv2.putText(
            img, text, position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
        )

    # Add FOV angle text at cardinal points
    positions = [
        (round(principal_point[0] + distance_to_fov), round(principal_point[1])),  # Right
        (round(principal_point[0] - distance_to_fov), round(principal_point[1])),  # Left
        (round(principal_point[0]), round(principal_point[1] - distance_to_fov)),  # Top
        (round(principal_point[0]), round(principal_point[1] + distance_to_fov))   # Bottom
    ]

    for pos in positions:
        image = add_fov_text(image, str(fov_angle), pos)

    # Add FOV angle text at diagonal points
    diagonal_offset = distance_to_fov / np.sqrt(2)
    diagonal_positions = [
        (round(principal_point[0] + diagonal_offset),
         round(principal_point[1] + diagonal_offset)),  # Bottom-right
        (round(principal_point[0] - diagonal_offset),
         round(principal_point[1] + diagonal_offset)),  # Bottom-left
        (round(principal_point[0] + diagonal_offset),
         round(principal_point[1] - diagonal_offset)),  # Top-right
        (round(principal_point[0] - diagonal_offset),
         round(principal_point[1] - diagonal_offset))   # Top-left
    ]

    for pos in diagonal_positions:
        image = add_fov_text(image, str(fov_angle), pos)

    # Save and display results
    output_path = os.path.join(PATHS["debug_data"], 'fov_test.jpg')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

    print(f"{Fore.GREEN}Estimated FOV is {fov_angle}°{Style.RESET_ALL}")
    print(f"{Fore.LIGHTCYAN_EX}Please verify that the FOV circle aligns with your fisheye view border!")
    print(f"The image has been saved to: {output_path}{Style.RESET_ALL}")


def estimate_fov_optimized(
    poly_incident_angle_to_radius: List[float],
    principal_point: List[float]
) -> Optional[float]:
    """
    Estimate the optimal field of view angle using optimization.

    Uses Nelder-Mead optimization to find the FOV angle that maximizes
    the distance from the principal point to the projected point.

    Args:
        poly_incident_angle_to_radius: Polynomial coefficients for angle to radius mapping
        principal_point: Image center coordinates [x, y]

    Returns:
        Estimated FOV angle in degrees, or None if optimization fails
    """
    def reprojection_error(fov_angle: float) -> float:
        """Calculate negative distance to maximize during optimization."""
        theta = np.deg2rad(fov_angle)
        x_prime = np.tan(theta)

        # Project to image coordinates
        projected = camera_coords_to_image_intrinsic(
            np.column_stack((x_prime, [0])),
            poly_incident_angle_to_radius,
            principal_point
        )

        # Return negative distance (we're minimizing)
        return -(projected[0][0] - principal_point[0])

    # Run optimization
    result = minimize(
        reprojection_error,
        x0=60,  # Initial guess
        method='Nelder-Mead',
        bounds=[(20, 90)]  # Constrain FOV between 20° and 90°
    )

    return result.x[0] if result.success else None


def import_camera_intrinsic_function() -> Tuple[List[float], List[float], Optional[float]]:
    """
    Import camera intrinsic parameters and FOV from calibration file.

    This function reads camera calibration parameters from a YAML file and estimates
    the field of view if not already available. The estimated FOV is then saved
    back to the calibration file and visualized for verification.

    Returns:
        Tuple containing:
        - poly_incident_angle_to_radius: Polynomial coefficients for angle to radius mapping
        - principal_point: Image center coordinates [x, y]
        - estimated_fov: Estimated field of view in degrees
    """
    calibration_file = 'calibration.yml'

    try:
        with open(calibration_file) as f:
            data = yaml.safe_load(f)
            poly_incident_angle_to_radius = data['poly_incident_angle_to_radius']
            principal_point = data['principal_point']
            estimated_fov = data.get('fov')
    except Exception as e:
        print(f"{Fore.RED}Failed to load calibration data: {e}{Style.RESET_ALL}")
        return None, None, None

    # Estimate FOV if not available
    if estimated_fov is None:
        print(f"{Fore.LIGHTCYAN_EX}Estimating FOV using optimization...{Style.RESET_ALL}")

        estimated_fov = estimate_fov_optimized(poly_incident_angle_to_radius, principal_point)

        if estimated_fov:
            # Save estimated FOV
            data['fov'] = estimated_fov.tolist()
            try:
                with open(calibration_file, 'w') as f:
                    yaml.safe_dump(data, f)
                print(f"{Fore.GREEN}FOV estimation saved to {calibration_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Failed to save FOV estimation: {e}{Style.RESET_ALL}")

            # Create visualization for verification
            create_image(estimated_fov, poly_incident_angle_to_radius, principal_point)
        else:
            print(f"{Fore.RED}FOV estimation failed{Style.RESET_ALL}")

    return poly_incident_angle_to_radius, principal_point, estimated_fov
