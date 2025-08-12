import numpy as np
from camera_coords_to_image_intrinsic import camera_coords_to_image_intrinsic
from astropy_to_camera_extrinsic import ground_homogeneous_to_camera_extrinsic
from colorama import Fore, Style
from typing import Tuple


def compute_diffuse_shading_factor(
    image: np.ndarray,
    poly_incident_angle_to_radius,
    principal_point: Tuple[float, float],
    estimated_fov: float,
    im_height: int,
    im_width: int,
    image_orientation: float,
    image_inclination: float,
    inclined_surface_orientation: float,
    inclined_surface_inclination: float
) -> float:
    """
    Computes the diffuse shading factor from a fisheye sky mask image.

    Args:
        image: Binary fisheye image (sky=255, obstacle=0)
        poly_incident_angle_to_radius: OmniCalib calibration object or mapping
        principal_point: (cx, cy) image center
        estimated_fov: Estimated field of view of the fisheye lens (degrees)
        im_height: Original image height
        im_width: Original image width
        image_orientation: Azimuth of the camera (degrees)
        image_inclination: Tilt of the camera from vertical (degrees)
        inclined_surface_orientation: Orientation of the solar surface (degrees)
        inclined_surface_inclination: Inclination of the solar surface from horizontal (degrees)

    Returns:
        Diffuse shading factor (0: fully visible, 1: fully shaded)
    """
    print(f"{Fore.YELLOW}Computing global DIFFUSE shading factor...{Style.RESET_ALL}")
    estimated_fov_rad = np.radians(estimated_fov)

    # Output image dimensions for remapping
    azimuth_length = 1000
    zenith_length = 500

    # Mapping constants
    k1 = zenith_length / (1 - np.cos(estimated_fov_rad))
    k2 = azimuth_length / (2 * np.pi)

    # Create index grid for remapped image
    hor_idx = np.arange(azimuth_length)
    ver_idx = np.arange(zenith_length)
    azimuth_grid = hor_idx / k2
    zenith_grid = np.arccos(1 - (ver_idx + 1) / k1)

    zenith_mat, azimuth_mat = np.meshgrid(zenith_grid, azimuth_grid, indexing='ij')

    # Convert spherical to camera-centered Cartesian coordinates
    x_prime = np.cos(azimuth_mat) * np.tan(zenith_mat)
    y_prime = np.sin(azimuth_mat) * np.tan(zenith_mat)

    # Stack into shape (zenith_length, azimuth_length, 2)
    coords_cam = np.stack([x_prime, y_prime], axis=2)

    # Map camera coords to image pixel coords using camera calibration
    # This must return an array of shape (zenith_length, azimuth_length, 2)
    equi_points = camera_coords_to_image_intrinsic(
        coords_cam.reshape(-1, 2).tolist(),
        poly_incident_angle_to_radius,
        principal_point
    )
    equi_points = np.array(equi_points).reshape((zenith_length, azimuth_length, 2)).astype(int)

    # Create equiareal image by sampling the original fisheye binary mask
    valid_mask = (
        (equi_points[:, :, 0] >= 0) & (equi_points[:, :, 0] < im_width) &
        (equi_points[:, :, 1] >= 0) & (equi_points[:, :, 1] < im_height)
    )

    equiareal_image = np.zeros((zenith_length, azimuth_length), dtype=np.uint8)
    equiareal_image[valid_mask] = image[
        equi_points[:, :, 1][valid_mask],
        equi_points[:, :, 0][valid_mask]
    ]

    # Compute normal vector to inclined surface (in ground coordinates)
    azimuth_panel_rad = np.radians(inclined_surface_orientation + 90)
    zenith_panel_rad = np.radians(inclined_surface_inclination)

    x_ground = np.sin(zenith_panel_rad) * np.cos(azimuth_panel_rad) / np.cos(zenith_panel_rad)
    y_ground = np.sin(zenith_panel_rad) * np.sin(azimuth_panel_rad) / np.cos(zenith_panel_rad)

    # Convert to camera extrinsic coordinates
    x_cam, y_cam = ground_homogeneous_to_camera_extrinsic(
        [x_ground, y_ground], image_orientation, image_inclination
    )

    # Generate 3D vectors of sky directions
    x_vec = x_prime
    y_vec = y_prime
    z_vec = np.ones_like(x_vec)

    # Dot product with plane normal: X*Nx + Y*Ny + Z*Nz > 0
    visible_mask = (x_vec * x_cam + y_vec * y_cam + z_vec) > 0

    # Mask out non-visible sky regions
    equiareal_image[~visible_mask] = 0

    # Total possible diffuse area (i.e., area visible to the panel)
    total_area = np.sum(visible_mask)

    # Unshaded area (white pixels only, i.e., sky = 255)
    visible_sky_area = np.sum(equiareal_image == 255)

    # Compute diffuse shading factor
    if total_area > 0:
        shading_factor = 1 - (visible_sky_area / total_area)
    else:
        shading_factor = 1.0  # fully blocked

    print(f"{Fore.GREEN}Diffuse shading factor computed: {shading_factor:.2f}{Style.RESET_ALL}")
    return shading_factor
