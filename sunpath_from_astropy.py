import os
from typing import Tuple
import numpy as np
import pandas as pd
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from colorama import Fore, Style
from config import PATHS


def sunpath_from_astropy(
    longitude: float,
    latitude: float,
    ground_level: float,
    time_array: pd.DatetimeIndex
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate solar position over time using Astropy.

    Args:
        longitude: Site longitude in degrees (-180 to 180)
        latitude: Site latitude in degrees (-90 to 90)
        ground_level: Site elevation above sea level in meters
        time_array: Array of timestamps for calculation

    Returns:
        Tuple containing:
        - np.ndarray: Azimuth angles in radians
        - np.ndarray: Zenith angles in radians

    Raises:
        ValueError: If coordinates are invalid
    """
    # Validate inputs
    if not (-90 <= latitude <= 90):
        raise ValueError(
            f"{Fore.RED}Invalid latitude: {latitude}. Must be between -90 and 90.{Style.RESET_ALL}")
    if not (-180 <= longitude <= 180):
        raise ValueError(
            f"{Fore.RED}Invalid longitude: {longitude}. Must be between -180 and 180.{Style.RESET_ALL}")

    print(f'{Fore.YELLOW}Calculating solar trajectory using Astropy...{Style.RESET_ALL}')

    try:
        # Convert timestamps to Astropy time objects
        astropy_time = Time(time_array)

        # Create location object with proper units
        deployment_location = coord.EarthLocation(
            lat=latitude * u.deg,
            lon=longitude * u.deg,
            height=ground_level * u.m
        )

        # Calculate sun position in horizontal coordinates
        frame_prediction = coord.AltAz(
            obstime=astropy_time,
            location=deployment_location
        )
        sun_pos_array = coord.get_sun(astropy_time).transform_to(frame_prediction)

        # Extract zenith and azimuth values
        zen_array = sun_pos_array.zen.value  # Angle from vertical
        az_array = sun_pos_array.az.value    # Angle from North (clockwise)

        # Save results for debugging and verification
        debug_data = pd.DataFrame({
            'Timeseries': time_array,
            'Zen_array': zen_array,
            'Az_array': az_array
        })

        # Save to debug directory
        output_path = os.path.join(PATHS["debug_data"], 'solar_coords.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        debug_data.set_index('Timeseries').to_csv(output_path)

        print(f'{Fore.GREEN}Solar trajectory calculation complete!{Style.RESET_ALL}')
        return [az_array, zen_array]

    except Exception as e:
        print(f"{Fore.RED}Error calculating solar trajectory: {str(e)}{Style.RESET_ALL}")
        raise
