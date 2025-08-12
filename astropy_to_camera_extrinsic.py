from typing import List, Union
import numpy as np


def degrees_to_radians(angles: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
    """Convert angles from degrees to radians.

    Args:
        angles: Single angle or array of angles in degrees

    Returns:
        Converted angle(s) in radians
    """
    return np.array(angles) * np.pi / 180


def astropy_to_camera_extrinsic(
    astropy_coords: List[float],
    psi: float,
    omega: float
) -> List[float]:
    """
    Convert astronomical coordinates to camera frame coordinates.

    Args:
        astropy_coords: List containing [azimuth, zenith] angles in degrees
            - azimuth: Solar azimuth angle (0° = North, 90° = East)
            - zenith: Solar zenith angle (0° = Zenith, 90° = Horizon)
        psi: Camera orientation angle in degrees (rotation around vertical axis)
        omega: Camera inclination angle in degrees (tilt from vertical)

    Returns:
        List containing [x, y] coordinates in the camera frame

    Note:
        The transformation follows these steps:
        1. Convert angles to radians
        2. Transform from astronomical to ground spherical coordinates
        3. Convert to ground rectangular coordinates
        4. Apply camera orientation rotation
        5. Apply camera inclination transformation
    """
    # Convert all angles to radians
    theta = degrees_to_radians(astropy_coords[0])  # azimuth
    phi = degrees_to_radians(astropy_coords[1])    # zenith
    psi = degrees_to_radians(psi)                  # orientation
    omega = degrees_to_radians(omega)              # inclination

    # Transform from astronomical to ground spherical coordinates
    # Astronomical: 0° = North, Ground: 0° = South
    theta_ground = 3 * np.pi / 2 - theta
    phi_ground = phi  # Zenith angle remains the same

    try:
        # Convert spherical ground coordinates to rectangular ground coordinates
        cos_phi = np.cos(phi_ground)
        if np.any(np.abs(cos_phi) < 1e-6):
            # For points very close to horizon, use small offset to avoid division by zero
            cos_phi = np.where(np.abs(cos_phi) < 1e-6, 1e-6 * np.sign(cos_phi), cos_phi)

        x_ground = np.sin(phi_ground) * np.cos(theta_ground) / cos_phi
        y_ground = np.sin(phi_ground) * np.sin(theta_ground) / cos_phi

        # Apply camera orientation rotation (around Z axis)
        x_rotated = np.cos(psi) * x_ground + np.sin(psi) * y_ground
        y_rotated = -np.sin(psi) * x_ground + np.cos(psi) * y_ground

        # Transform to camera coordinates with perspective division
        denominator = np.sin(omega) * y_rotated + np.cos(omega)
        small_denom = np.abs(denominator) < 1e-6

        if np.any(small_denom):
            # For points where denominator is close to zero, use small offset
            denominator = np.where(small_denom, 1e-6 * np.sign(denominator), denominator)

        x_camera = x_rotated / denominator
        y_camera = (np.cos(omega) * y_rotated - np.sin(omega)) / denominator

        return np.array([x_camera, y_camera]).T.tolist()

    except Exception as e:
        raise ValueError(f"AstroPy to camera transformation failed: {str(e)}")


def ground_homogeneous_to_camera_extrinsic(
    ground_coords: List[float],
    psi: float,
    omega: float
) -> List[float]:
    """
    Transform ground homogeneous coordinates to camera extrinsic coordinates.

    Args:
        ground_coords: List containing [x, y] ground homogeneous coordinates
        psi: Camera orientation angle in degrees (rotation around vertical axis)
        omega: Camera inclination angle in degrees (tilt from vertical)

    Returns:
        List containing [x, y] coordinates in the camera frame

    Note:
        The transformation follows these steps:
        1. Convert angles to radians
        2. Apply camera orientation rotation
        3. Apply camera inclination transformation with perspective division
    """
    try:
        # Convert angles to radians
        psi_rad = degrees_to_radians(psi)
        omega_rad = degrees_to_radians(omega)

        x_ground, y_ground = ground_coords[0], ground_coords[1]

        # Apply camera orientation rotation (around Z axis)
        x_rotated = np.cos(psi_rad) * x_ground + np.sin(psi_rad) * y_ground
        y_rotated = -np.sin(psi_rad) * x_ground + np.cos(psi_rad) * y_ground

        # Transform to camera coordinates with perspective division
        denominator = np.sin(omega_rad) * y_rotated + np.cos(omega_rad)
        small_denom = np.abs(denominator) < 1e-6

        if np.any(small_denom):
            # For points where denominator is close to zero, use small offset
            denominator = np.where(small_denom, 1e-6 * np.sign(denominator), denominator)

        x_camera = x_rotated / denominator
        y_camera = (np.cos(omega_rad) * y_rotated - np.sin(omega_rad)) / denominator

        return np.array([x_camera, y_camera]).T.tolist()

    except Exception as e:
        raise ValueError(f"Ground to camera transformation failed: {str(e)}")
