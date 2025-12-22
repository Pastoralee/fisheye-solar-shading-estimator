from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import cv2
import os
from colorama import Fore, Style
from config import PATHS
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


def detect_isolated_outliers(
    image_coords: np.ndarray
) -> np.ndarray:
    """
    Detect isolated outliers by checking if both neighboring points are far away.
    Uses trajectory-based threshold estimation rather than fixed image dimensions.
    
    Args:
        image_coords: Array of projected image coordinates (N, 2)
        
    Returns:
        np.ndarray: Boolean mask where True indicates outlier positions
    """
    n_points = len(image_coords)
    outlier_mask = np.zeros(n_points, dtype=bool)
    
    if n_points < 3:
        return outlier_mask
    
    # Calculate all consecutive distances to estimate normal trajectory spacing
    consecutive_distances = []
    for i in range(1, n_points):
        dist = np.sqrt(np.sum((image_coords[i] - image_coords[i-1])**2))
        consecutive_distances.append(dist)
    
    consecutive_distances = np.array(consecutive_distances)
    
    # Use median distance as robust estimate of normal trajectory spacing
    median_distance = np.median(consecutive_distances)
    
    # Set threshold based on trajectory characteristics
    distance_threshold = 3 * median_distance
    outliers_detected = 0
    
    # Check each point, including first and last
    for i in range(n_points):
        current_point = image_coords[i]
        
        if i == 0:
            # First point: only check distance to next point
            if n_points > 1:
                next_point = image_coords[i + 1]
                dist_to_next = np.sqrt(np.sum((current_point - next_point)**2))
                if dist_to_next > distance_threshold:
                    outlier_mask[i] = True
                    outliers_detected += 1
        
        elif i == n_points - 1:
            # Last point: only check distance to previous point
            prev_point = image_coords[i - 1]
            dist_to_prev = np.sqrt(np.sum((current_point - prev_point)**2))
            if dist_to_prev > distance_threshold:
                outlier_mask[i] = True
                outliers_detected += 1
        
        else:
            # Middle points: check distances to both neighbors
            prev_point = image_coords[i - 1]
            next_point = image_coords[i + 1]
            
            # Calculate distances to neighbors
            dist_to_prev = np.sqrt(np.sum((current_point - prev_point)**2))
            dist_to_next = np.sqrt(np.sum((current_point - next_point)**2))
            
            # If both neighbors are far away, mark as outlier
            if dist_to_prev > distance_threshold and dist_to_next > distance_threshold:
                outlier_mask[i] = True
                outliers_detected += 1
    
    print(f"{Fore.YELLOW}Trajectory-based outlier detection: "
          f"Found {outliers_detected} isolated outliers ({outliers_detected/n_points*100:.1f}%){Style.RESET_ALL}")

    return outlier_mask


def compute_outliers_robust(
    disk_radii: np.ndarray,
    k_factor: float = 3.0,
    epsilon: float = 0.1
) -> np.ndarray:
    """
    Detect outliers based on simple criteria:
    - Values that are too high (beyond median + k_factor * MAD)
    - Values that are too low (below median - k_factor * MAD)
    - Extremely low values (under 1 pixel)
    
    Args:
        disk_radii: Array of disk radius measurements
        k_factor: Threshold factor for MAD-based outlier detection (default 3.0)
        epsilon: Minimum relative threshold as fraction of median (default 0.1)
    
    Returns:
        outlier_mask: Boolean array where True indicates outlier
    """
    if len(disk_radii) == 0:
        return np.array([], dtype=bool)
    
    # First, detect extremely low values
    extremely_low = disk_radii <= 1.0
    
    # Filter out extremely low values before computing robust statistics
    valid_radii = disk_radii[~extremely_low]
    
    # If all values are extremely low, mark all as outliers
    if len(valid_radii) == 0:
        return np.array([], dtype=bool)
    
    # Calculate robust statistics using median and MAD on valid values
    median_val = np.median(valid_radii)
    mad = np.median(np.abs(valid_radii - median_val))
    
    # Apply robust scale factor (MAD to standard deviation conversion)
    robust_scale = 1.4826 * mad
    
    # Set threshold with minimum floor to prevent scale collapse
    threshold = max(k_factor * robust_scale, epsilon * median_val)
    
    # Detect outliers:
    # 1. Values too high (beyond upper threshold)
    too_high = disk_radii > (median_val + threshold)
    
    # 2. Values too low
    if median_val > threshold:
        too_low = disk_radii < (median_val - threshold)
    else:
        too_low = disk_radii < median_val * 0.33
    
    # Combine all outlier conditions
    outlier_mask = too_high | too_low | extremely_low
    
    return outlier_mask


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
        ze = np.degrees(np.arccos(np.clip(v[2], -1, 1)))
        az = np.degrees(np.arctan2(v[0], v[1])) % 360
        disk_az_ze.append([az, ze])

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

    # Use robust outlier detection
    outlier_mask = compute_outliers_robust(
        radii,
        k_factor=3.0,
        epsilon=0.1
    )
    good_radii = radii[~outlier_mask]
    if len(good_radii) == 0:
        return np.median(radii)
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

    # Detect and remove isolated outliers
    outlier_mask = detect_isolated_outliers(image_coords)
    image_coords = image_coords[~outlier_mask]
    valid_indices = valid_indices[~outlier_mask]
    
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
        radius_px = compute_sun_disk_radius_pixels(
            az_array[i],
            zen_array[i],
            image_orientation,
            image_inclination,
            poly_incident_angle_to_radius,
            principal_point,
        )
        radius_px = round(radius_px)

        # Draw sun path and disk
        path_thickness = int(max(1, radius_px * 2))
        cv2.line(mask_im, pt1, pt2, 255, path_thickness)
        if radius_px > 0:
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
    debug_path = os.path.join(PATHS["debug_data"], 'sun_trajectory.jpg')
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    cv2.imwrite(debug_path, trajectory_image)
    print(f"{Fore.GREEN}Saved trajectory visualization to {debug_path}{Style.RESET_ALL}")

    # Calculate final factors
    shading_factors = 1 - np.multiply(
        complementary_direct_shading_factor,
        plane_adjusted_coeff
    )

    print(f"{Fore.GREEN}Done computing direct shading factors!{Style.RESET_ALL}")
    return shading_factors
