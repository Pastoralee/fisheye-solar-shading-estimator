from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import cv2
from colorama import Fore, Style
from camera_coords_to_image_intrinsic import camera_coords_to_image_intrinsic
from astropy_to_camera_extrinsic import astropy_to_camera_extrinsic


def angular_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the angular distance between two vectors in degrees.

    Args:
        vec1: First vector as numpy array
        vec2: Second vector as numpy array

    Returns:
        float: Angular distance in degrees between the vectors

    Note:
        Vectors are normalized before calculation to ensure accurate angle measurement.
    """
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def compute_sun_disk_radius(
    azimuth_deg: float,
    zenith_deg: float,
    psi: float,
    omega: float,
    principal_point: np.ndarray,
    poly_incident_angle_to_radius: np.ndarray,
    include_circumsolar: bool = False
) -> float:
    """
    Estimate the apparent pixel radius of the sun's disk in a fisheye image.

    Args:
        azimuth_deg: Solar azimuth angle in degrees
        zenith_deg: Solar zenith angle in degrees
        psi: Image orientation angle in degrees
        omega: Image inclination angle in degrees
        principal_point: Image principal point coordinates [x, y]
        poly_incident_angle_to_radius: Polynomial coefficients for angle-to-radius mapping
        include_circumsolar: Whether to include circumsolar region (default: False)

    Returns:
        float: Estimated radius of the sun disk in pixels

    Raises:
        ValueError: If sun disk radius calculation fails due to invalid angles

    Note:
        When include_circumsolar is True, uses 2.5° for radius instead of 0.53°/2
    """
    # Angular radius of the sun
    if include_circumsolar:
        sun_radius_angle_deg = 2.5
    else:
        sun_radius_angle_deg = 0.53 / 2

    # Central vector (sun center)
    center_vec = astropy_to_camera_extrinsic([azimuth_deg, zenith_deg], psi, omega)
    sun_center = camera_coords_to_image_intrinsic(
        center_vec, poly_incident_angle_to_radius, principal_point)

    # Offset vectors (sun edge), just slightly more or less zenith (i.e., toward horizon)
    # We compute both the positive and negative offsets to ensure we capture
    # the sun's disk correctly
    positive_offset_edge_vec = astropy_to_camera_extrinsic(
        [azimuth_deg, zenith_deg + sun_radius_angle_deg], psi, omega
    )
    positive_offset_edge = camera_coords_to_image_intrinsic(
        positive_offset_edge_vec, poly_incident_angle_to_radius, principal_point
    )
    sun_distance_to_positive_edge = np.linalg.norm(positive_offset_edge - sun_center)

    negative_offset_edge_vec = astropy_to_camera_extrinsic(
        [azimuth_deg, zenith_deg - sun_radius_angle_deg], psi, omega
    )
    negative_offset_edge = camera_coords_to_image_intrinsic(
        negative_offset_edge_vec, poly_incident_angle_to_radius, principal_point
    )
    sun_distance_to_negative_edge = np.linalg.norm(negative_offset_edge - sun_center)

    angle_pos = angular_distance(center_vec, positive_offset_edge_vec)
    angle_neg = angular_distance(center_vec, negative_offset_edge_vec)
    # For circumsolar regions, we need more lenient validation
    # Allow up to 3x the expected angle to account for projection distortions
    max_allowed_angle = sun_radius_angle_deg * 3
    ok_pos = 0 < angle_pos < max_allowed_angle
    ok_neg = 0 < angle_neg < max_allowed_angle

    if ok_pos and ok_neg:
        return max(sun_distance_to_positive_edge, sun_distance_to_negative_edge)
    elif ok_pos:
        return sun_distance_to_positive_edge
    elif ok_neg:
        return sun_distance_to_negative_edge
    else:
        raise ValueError(
            f"{Fore.RED}Sun disk radius calculation failed for azimuth={azimuth_deg}, zenith={zenith_deg}. "
            f"Detected Angles: pos={angle_pos:.3f}, neg={angle_neg:.3f}. "
            f"Distances: pos={sun_distance_to_positive_edge:.3f}, neg={sun_distance_to_negative_edge:.3f}.{Style.RESET_ALL}")


def compute_sun_disk_radius_pixels(
    azimuth_deg: float,
    zenith_deg: float,
    psi: float,
    omega: float,
    poly_incident_angle_to_radius: List[float],
    principal_point: List[float],
    sun_angle_deg: float = 2.5,
    num_samples: int = 16
) -> Tuple[np.ndarray, float]:
    """
    Computes the position and pixel radius of the Sun disk (with circumsolar region) in a fisheye image.

    Args:
        azimuth_deg: Solar azimuth angle in degrees
        zenith_deg: Solar zenith angle in degrees
        psi: Camera orientation angle in degrees
        omega: Camera inclination angle in degrees
        poly_incident_angle_to_radius: Polynomial coefficients for fisheye mapping
        principal_point: [x, y] image center
        sun_angle_deg: Angular radius of the Sun disk (default 2.5° for circumsolar)
        num_samples: Number of points to sample around the Sun disk edge

    Returns:
        sun_center_img: [x, y] image coordinates of the Sun center
        radius_pix: Estimated pixel radius of the Sun disk
    """

    # Convert Sun direction to 3D unit vector
    az = np.radians(azimuth_deg)
    ze = np.radians(zenith_deg)
    sun_vec = np.array([
        np.sin(ze) * np.sin(az),
        np.sin(ze) * np.cos(az),
        np.cos(ze)
    ])

    # Build two perpendicular vectors to sun_vec
    z_axis = np.array([0, 0, 1])
    if np.allclose(sun_vec, z_axis):
        ortho1 = np.array([1, 0, 0])
    else:
        ortho1 = np.cross(sun_vec, z_axis)
        ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(sun_vec, ortho1)

    # Sample disk edge on the sphere
    alpha = np.radians(sun_angle_deg)
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    disk_vectors = [
        np.cos(alpha) * sun_vec + np.sin(alpha) * (np.cos(a) * ortho1 + np.sin(a) * ortho2)
        for a in angles
    ]

    # Convert vectors to azimuth/zenith
    disk_az_ze = []
    for v in disk_vectors:
        v = v / np.linalg.norm(v)
        ze = np.degrees(np.arccos(v[2]))
        az = np.degrees(np.arctan2(v[0], v[1])) % 360
        disk_az_ze.append([az, ze])

    # print(f"disk_vectors: {disk_vectors}")
    # print(f"disk_az_ze: {disk_az_ze}")

    # Convert disk points to image coordinates
    disk_cam_coords = [
        astropy_to_camera_extrinsic([az, ze], psi, omega)
        for az, ze in disk_az_ze
    ]
    disk_img_coords = camera_coords_to_image_intrinsic(
        disk_cam_coords, poly_incident_angle_to_radius, principal_point
    )

    # Project Sun center
    sun_center_cam = astropy_to_camera_extrinsic([azimuth_deg, zenith_deg], psi, omega)
    sun_center_img = camera_coords_to_image_intrinsic(
        [sun_center_cam], poly_incident_angle_to_radius, principal_point
    )[0]

    # Compute average distance to edge points
    radii = [np.linalg.norm(pt - sun_center_img) for pt in disk_img_coords]
    radii = np.array(radii)

    # print(f"disk_img_coords: {disk_img_coords}")
    # print(f"sun_center_img: {sun_center_img}")
    # print(f"radii: {radii}")

    # Use median + MAD (median absolute deviation)
    median = np.median(radii)
    mad = np.median(np.abs(radii - median))

    # Keep only values within a robust range (e.g., 3 * MAD)
    good_radii = radii[np.abs(radii - median) < 3 * mad]

    if len(good_radii) == 0:
        return median  # fallback to median

    return float(np.mean(good_radii))


def compute_irradiance_projection_coeff(
    az_zen_array: np.ndarray,
    irradiance_type: str,
    surf_azimuth: Optional[float],
    surf_tilt: Optional[float]
) -> np.ndarray:
    """
    Calculate the projection coefficient for irradiance on a surface.

    Args:
        az_zen_array: Solar position [azimuth, zenith] arrays
        irradiance_type: Type of irradiance data ('normal' or 'horizontal')
        surf_azimuth: Surface azimuth angle in degrees
        surf_tilt: Surface tilt angle in degrees

    Returns:
        np.ndarray: Coefficient array for adjusting irradiance values

    Note:
        For 'normal' irradiance (DNI): Projects the direct normal irradiance onto the inclined surface
        For 'horizontal' irradiance (BHI): First gets the normal component, then projects onto inclined surface
    """
    zenith = np.radians(az_zen_array[1])
    azimuth = np.radians(270 - az_zen_array[0])  # Convert to local coord.
    tilt_rad = np.radians(surf_tilt)
    az_surf_rad = np.radians(90 + surf_azimuth)

    if irradiance_type == 'normal':
        # For DNI: Direct projection onto inclined surface
        coeff = (
            np.cos(tilt_rad) * np.cos(zenith) +
            np.sin(tilt_rad) * np.sin(zenith) * np.cos(azimuth - az_surf_rad)
        )
    elif irradiance_type == 'horizontal':
        # For BHI: First get normal component (divide by cos(zenith)),
        # then project onto inclined surface
        cos_zen = np.cos(zenith)
        dni = np.zeros_like(cos_zen)
        valid = cos_zen > 0
        dni[valid] = 1.0 / cos_zen[valid]
        coeff = dni * (
            np.cos(tilt_rad) * np.cos(zenith) +
            np.sin(tilt_rad) * np.sin(zenith) * np.cos(azimuth - az_surf_rad)
        )

    coeff = np.clip(coeff, 0, 1)
    return coeff


def compute_direct_shading_factor_generic(
    image: np.ndarray,
    im_height: int,
    im_width: int,
    poly_incident_angle_to_radius: np.ndarray,
    principal_point: np.ndarray,
    image_orientation: float,
    image_inclination: float,
    estimated_fov: float,
    az_zen_array: np.ndarray,
    original_time_array: np.ndarray,
    inclined_surface_orientation: Optional[float] = None,
    inclined_surface_inclination: Optional[float] = None,
    irradiance_type: str = 'horizontal'
) -> np.ndarray:
    """
    Generic direct shading factor calculator for any irradiance data source.

    Args:
        image: Binary mask image of sky/obstacles
        im_height: Image height in pixels
        im_width: Image width in pixels
        poly_incident_angle_to_radius: Polynomial coefficients for angle-to-radius mapping
        principal_point: Image principal point coordinates [x, y]
        image_orientation: Camera orientation angle in degrees
        image_inclination: Camera inclination angle in degrees
        estimated_fov: Estimated field of view in degrees
        az_zen_array: Array of [azimuth, zenith] angles
        original_time_array: Array of timestamps
        inclined_surface_orientation: Inclined surface azimuth angle in degrees
        inclined_surface_inclination: Inclined surface inclination angle in degrees
        irradiance_type: Type of input irradiance data:
            - 'normal': Direct normal irradiance (like NASA POWER)
            - 'horizontal': Direct horizontal irradiance

    Returns:
        np.ndarray: Array of shading factors (0-1) for each timestamp
    """
    print(f"{Fore.YELLOW}Computing direct shading factors for {irradiance_type} irradiance...{Style.RESET_ALL}")

    # Initialize arrays
    complementary_direct_shading_factor = np.zeros(len(original_time_array))

    # Calculate surface adjustment if needed
    plane_adjusted_coeff = compute_irradiance_projection_coeff(
        az_zen_array,
        irradiance_type,
        inclined_surface_orientation,
        inclined_surface_inclination
    )

    # Filter points below horizon
    valid_mask = az_zen_array[1] <= estimated_fov
    valid_indices = np.where(valid_mask)[0]
    az_array = az_zen_array[0][valid_mask]
    zen_array = az_zen_array[1][valid_mask]

    # Convert solar positions to image coordinates
    camera_homo_coords = astropy_to_camera_extrinsic(
        [az_array, zen_array],
        image_orientation,
        image_inclination
    )
    image_coords = camera_coords_to_image_intrinsic(
        camera_homo_coords,
        poly_incident_angle_to_radius,
        principal_point
    )

    # Process image and create visualization
    image = image.astype(np.uint8)
    trajectory_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    # Process each sun position
    for i in tqdm(range(1, len(valid_indices)), desc="Processing sun positions"):
        if (valid_indices[i] - valid_indices[i - 1] != 1):
            continue
        idx = valid_indices[i]

        # Create mask for sun position
        mask_im = np.zeros(shape=(im_height, im_width, 1), dtype=np.uint8)
        pt1 = tuple(map(int, image_coords[i - 1]))
        pt2 = tuple(map(int, image_coords[i]))

        # Calculate sun disk radius
        # radius_px = round(compute_sun_disk_radius(
        #     az_array[i],
        #     zen_array[i],
        #     image_orientation,
        #     image_inclination,
        #     principal_point,
        #     poly_incident_angle_to_radius,
        #     include_circumsolar=True
        # ))

        radius_px = round(compute_sun_disk_radius_pixels(
            az_array[i],
            zen_array[i],
            image_orientation,
            image_inclination,
            poly_incident_angle_to_radius,
            principal_point,
        ))

        # Draw sun path and disk
        path_thickness = radius_px * 2
        cv2.line(mask_im, pt1, pt2, 255, path_thickness)
        cv2.circle(mask_im, pt2, radius_px, 255, -1)

        # Draw visualization
        cv2.line(trajectory_image, pt1, pt2, (0, 0, 255), path_thickness)

        # Calculate shading factor
        if cv2.countNonZero(mask_im) > 0:
            masked_im = cv2.bitwise_and(image, image, mask=mask_im)
            visible_pixels = cv2.countNonZero(masked_im)
            total_pixels = cv2.countNonZero(mask_im)
            complementary_direct_shading_factor[idx] = visible_pixels / total_pixels

    # Save visualization
    debug_path = './DebugData/sun_trajectory.jpg'
    cv2.imwrite(debug_path, trajectory_image)
    print(f"{Fore.GREEN}Saved trajectory visualization to {debug_path}{Style.RESET_ALL}")

    # Calculate final factors
    shading_factors = 1 - np.multiply(
        complementary_direct_shading_factor,
        plane_adjusted_coeff
    )

    print(f"{Fore.GREEN}Done computing direct shading factors!{Style.RESET_ALL}")
    return shading_factors
