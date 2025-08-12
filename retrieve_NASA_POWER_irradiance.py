import json
from typing import Tuple, Union
import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from colorama import Fore, Style
from requests.exceptions import RequestException

# API Configuration
NASA_POWER_API = {
    "base_url": "https://power.larc.nasa.gov/api/temporal/hourly/point",
    "parameters": ["ALLSKY_SFC_SW_DIFF", "ALLSKY_SFC_SW_DNI"],
    "timeout": 30.0,
    "community": "RE",
    "format": "JSON",
    "time_standard": "UTC"
}


def retrieve_NASA_POWER_irradiance(
    lat: float,
    long: float,
    start_time: Union[str, int],
    end_time: Union[str, int]
) -> Tuple[NDArray, NDArray, pd.DatetimeIndex]:
    """
    Retrieve solar irradiance data from NASA POWER API.

    Fetches both direct normal irradiance (DNI) and diffuse horizontal
    irradiance (DHI) data for the specified location and time period.

    Args:
        lat: Latitude of the location (-90 to 90)
        long: Longitude of the location (-180 to 180)
        start_time: Start date in YYYYMMDD format
        end_time: End date in YYYYMMDD format

    Returns:
        Tuple containing:
        - NDArray: Direct normal irradiance (W/m²)
        - NDArray: Diffuse horizontal irradiance (W/m²)
        - DatetimeIndex: Hourly timestamps for the data

    Raises:
        RequestException: If API request fails
        ValueError: If coordinates are invalid or data parsing fails
    """
    # Validate inputs
    if not (-90 <= lat <= 90):
        raise ValueError(
            f"{Fore.RED}Invalid latitude: {lat}. Must be between -90 and 90.{Style.RESET_ALL}")
    if not (-180 <= long <= 180):
        raise ValueError(
            f"{Fore.RED}Invalid longitude: {long}. Must be between -180 and 180.{Style.RESET_ALL}")

    # Construct API URL
    api_request_url = (
        f"{NASA_POWER_API['base_url']}"
        f"?parameters={','.join(NASA_POWER_API['parameters'])}"
        f"&community={NASA_POWER_API['community']}"
        f"&longitude={long}&latitude={lat}"
        f"&start={start_time}&end={end_time}"
        f"&format={NASA_POWER_API['format']}"
        f"&time-standard={NASA_POWER_API['time_standard']}"
    )

    try:
        # Make API request
        print(f"{Fore.YELLOW}Fetching data from NASA POWER API...{Style.RESET_ALL}")
        response = requests.get(
            url=api_request_url,
            verify=True,
            timeout=NASA_POWER_API['timeout']
        )
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse response
        content = json.loads(response.content.decode('utf-8'))
        records = content['properties']['parameter']

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(records)

        # Parse timestamps and extract data
        datetime_index = pd.to_datetime(df.index, format="%Y%m%d%H")
        direct_data = df['ALLSKY_SFC_SW_DNI']
        diffuse_data = df['ALLSKY_SFC_SW_DIFF']

        print(f"{Fore.GREEN}Successfully retrieved NASA POWER data{Style.RESET_ALL}")
        return direct_data.to_numpy(), diffuse_data.to_numpy(), datetime_index

    except RequestException as e:
        raise
    except (KeyError, ValueError) as e:
        print(f"{Fore.RED}Error parsing NASA POWER data: {str(e)}{Style.RESET_ALL}")
        raise ValueError("Failed to parse NASA POWER response") from e
