import requests
import pandas as pd
from colorama import Fore, Style
from openpyxl import load_workbook
from typing import Optional
from validation import get_user_choice


def get_elevation_from_api(lat: float, lon: float, dataset: str = "srtm30m") -> Optional[float]:
    """
    Fetch elevation data from OpenTopoData API.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        dataset: Elevation dataset to use (default: 'srtm30m')

    Returns:
        float: Elevation in meters, or None if request fails
    """
    url = f"https://api.opentopodata.org/v1/{dataset}?locations={lat},{lon}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        result = data.get("results", [{}])[0]
        elevation = result.get("elevation")

        return elevation if elevation is not None else None
    except Exception as e:
        print(f"{Fore.YELLOW}Failed to fetch elevation from {dataset}: {e}{Style.RESET_ALL}")
        return None


def update_elevation_in_excel(file_path: str, new_elevation: float) -> bool:
    """
    Update only the elevation value in cell C2 of Excel file while preserving all formatting.

    Args:
        file_path: Path to the Excel file
        new_elevation: New elevation value to write

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the workbook while preserving formatting
        workbook = load_workbook(file_path)
        sheet = workbook.active

        # Update only cell C2 (elevation value)
        sheet['C2'].value = new_elevation

        # Save the workbook
        workbook.save(file_path)
        return True

    except Exception as e:
        print(f"{Fore.RED}Error updating Excel file: {e}{Style.RESET_ALL}")
        return False


def verify_and_update_elevation(specs: pd.DataFrame, excel_file_path: str) -> pd.DataFrame:
    """
    Verify the current elevation value against API data and offer correction.

    Checks if elevation is within reasonable bounds and compares with API-fetched elevation.
    Asks user for confirmation if discrepancy found and updates both DataFrame and Excel file.

    Args:
        specs: DataFrame with system specifications
        excel_file_path: Path to the Excel file to update

    Returns:
        pd.DataFrame: Updated specifications (may have corrected elevation)
    """
    try:
        lat = float(specs['Lattitude (°)'][0])
        lon = float(specs['Longitude (°)'][0])
        current_elevation = float(specs['Elevation (m)'][0])

        print(f"{Fore.YELLOW}Verifying elevation for coordinates ({lat:.4f}, {lon:.4f})...{Style.RESET_ALL}")

        # Try multiple datasets for better coverage
        datasets = ["eudem25m", "srtm30m"]
        api_elevation = None

        for dataset in datasets:
            api_elevation = get_elevation_from_api(lat, lon, dataset)
            if api_elevation is not None:
                print(f"{Fore.GREEN}Found elevation data from {dataset}: {api_elevation:.1f}m{Style.RESET_ALL}")
                break

        if api_elevation is not None:
            # Check if current elevation is close to API elevation (within 10% tolerance)
            elevation_diff = abs(current_elevation - api_elevation)
            tolerance = max(api_elevation * 0.1, 10)  # 10% tolerance, minimum 10m

            if elevation_diff > tolerance:
                print(f"{Fore.RED}Elevation discrepancy detected:")
                print(f"  Current elevation: {current_elevation}m")
                print(f"  Estimated elevation: {api_elevation:.1f}m")
                print(f"  Difference: {elevation_diff:.1f}m{Style.RESET_ALL}")

                print(f"\n{Fore.CYAN}What would you like to do?{Style.RESET_ALL}")
                print("1. Use estimated elevation value (recommended)")
                print("2. Keep current elevation value")
                print("3. Enter a new elevation value manually")

                choice = get_user_choice("Enter choice (1-3): ", ["1", "2", "3"])

                if choice == "1":
                    # Convert column to float to avoid dtype warning
                    specs['Elevation (m)'] = specs['Elevation (m)'].astype(float)
                    specs.loc[0, 'Elevation (m)'] = api_elevation
                    new_elevation = api_elevation
                    print(f"{Fore.GREEN}Elevation updated to: {api_elevation:.1f}m{Style.RESET_ALL}")
                    
                    # Update Excel file
                    if update_elevation_in_excel(excel_file_path, new_elevation):
                        print(f"{Fore.GREEN}Excel file updated with new elevation{Style.RESET_ALL}")

                elif choice == "2":
                    print(f"{Fore.GREEN}Keeping current elevation: {current_elevation}m{Style.RESET_ALL}")

                elif choice == "3":
                    while True:
                        try:
                            manual_elevation = float(
                                input(f"{Fore.CYAN}Enter elevation manually (meters): {Style.RESET_ALL}"))
                            specs.loc[0, 'Elevation (m)'] = manual_elevation
                            new_elevation = manual_elevation
                            print(f"{Fore.GREEN}Elevation set to: {manual_elevation}m{Style.RESET_ALL}")
                            
                            # Update Excel file
                            if update_elevation_in_excel(excel_file_path, new_elevation):
                                print(f"{Fore.GREEN}Excel file updated with new elevation{Style.RESET_ALL}")
                            break
                        except ValueError:
                            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}Elevation looks good: {current_elevation}m (Estimated: {api_elevation:.1f}m){Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Could not estimate elevation for coordinates ({lat:.4f}, {lon:.4f}). "
                  f"Current value: {current_elevation}m{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error verifying elevation: {e}{Style.RESET_ALL}")

    return specs
