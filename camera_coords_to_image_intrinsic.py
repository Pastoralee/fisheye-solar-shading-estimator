from typing import List, Union
import numpy as np
import torch
from omnicalib.projection import project_poly_thetar


def camera_coords_to_image_intrinsic(
    camera_coords: Union[np.ndarray, List[List[float]]],
    poly_incident_angle_to_radius: List[float],
    principal_point: List[float]
) -> np.ndarray:
    """
    Project 3D camera coordinates to 2D image coordinates using fisheye lens model.

    This function projects camera coordinates to image coordinates using the polynomial
    fisheye lens distortion model. It assumes calibration parameters are already available
    and validated.

    Args:
        camera_coords: Array of shape (N, 2) containing camera coordinates [x_c, y_c]
        poly_incident_angle_to_radius: Polynomial coefficients for the fisheye lens model
        principal_point: Image center coordinates [x, y]

    Returns:
        NDArray: Array of shape (N, 2) containing projected image coordinates [x_img, y_img]

    Note:
        This function does not verify calibration.yml existence.
        The calling code must ensure calibration parameters are valid.
    """
    # Convert inputs to PyTorch tensors
    camera_homo_coords = torch.Tensor(camera_coords)

    # Add homogeneous coordinate (z=1)
    padded_camera_homo_coords = torch.nn.functional.pad(
        camera_homo_coords,
        pad=(0, 1),  # Add one column at the end
        mode="constant",
        value=1.0    # Set z coordinate to 1
    )

    # Project coordinates using polynomial model
    image_coordinates = project_poly_thetar(
        view_points=padded_camera_homo_coords,
        poly_theta=poly_incident_angle_to_radius,
        principal_point=torch.Tensor(principal_point),
        normed=False
    ).numpy().astype(int)

    return image_coordinates
