import os
import requests
from glob import glob
from typing import List, Tuple
import cv2
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import IntEnum
from colorama import Fore, Style
from openpyxl import load_workbook


@dataclass
class DataSourceInfo:
    """Track information about data sources used."""
    using_nasa: bool = False
    using_day_night: bool = False
    last_irradiance_file: str = ""


class ProcessingStage(IntEnum):
    """Enumeration of processing stages to determine what needs to be recalculated."""
    CALIBRATION = 0      # Calibration needs to be redone
    RAW_IRRADIANCE = 1   # Raw irradiance needs to be recalculated
    SHADING = 2         # Shading needs to be recalculated
    STATE_OF_CHARGE = 3  # State of charge estimation needs to be recalculated
    NO_CHANGES = 4      # No recalculation needed

    @classmethod
    def get_min_stage(cls, current: 'ProcessingStage', new: 'ProcessingStage') -> 'ProcessingStage':
        """Get the minimum stage between two stages, handling NO_CHANGES properly."""
        if current == cls.NO_CHANGES:
            return new
        if new == cls.NO_CHANGES:
            return current
        return min(current, new)


@dataclass
class SystemSpecifications:
    """Container for system specification parameters."""
    LOCATION_PARAMS = ['Lattitude (°)', 'Longitude (°)', 'Elevation (m)']
    IMAGE_PARAMS = ['Image orientation (°)', 'Image inclination (°)',
                    'Plane orientation (°)', 'Plane inclination (°)']
    CALIB_PARAMS = ['Calib vertex short', 'Calib vertex long', 'Calib square size (mm)']
    TIME_PARAMS = ['Start year', 'End year']
    SYSTEM_PARAMS = ['Solar panel peak wattage (W)', 'Converter efficiency (%)',
                     'Converter max power (W)', 'Charge efficiency (%)',
                     'Discharge efficiency (%)', 'Max SOC (%)', 'Min SOC (%)',
                     'Batt nominal capacity (Ah)', 'Batt nominal voltage (V)']

    @classmethod
    def get_all_params(cls) -> List[str]:
        """Get list of all system parameters."""
        return (cls.LOCATION_PARAMS + cls.IMAGE_PARAMS + cls.CALIB_PARAMS +
                cls.TIME_PARAMS + cls.SYSTEM_PARAMS)


def get_elevation_from_api(lat: float, lon: float, dataset: str = "srtm30m") -> float:
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


def verify_elevation(specs: pd.DataFrame) -> pd.DataFrame:
    """
    Verify the current elevation value against API data and offer correction.

    Checks if elevation is within reasonable bounds (0-9000m) and compares
    with API-fetched elevation. Asks user for confirmation if discrepancy found.

    Args:
        specs: DataFrame with system specifications

    Returns:
        pd.DataFrame: Updated specifications (may have corrected elevation)
    """
    elevation_changed = False
    try:
        lat = float(specs['Lattitude (°)'][0])
        lon = float(specs['Longitude (°)'][0])
        current_elevation = float(specs['Elevation (m)'][0])

        # Try to get elevation from API for comparison
        print(f"{Fore.YELLOW}Verifying elevation for coordinates "
              f"({lat:.4f}, {lon:.4f})...{Style.RESET_ALL}")

        # Try multiple datasets for better coverage
        datasets = ["eudem25m", "srtm30m"]
        api_elevation = None

        for dataset in datasets:
            api_elevation = get_elevation_from_api(lat, lon, dataset)
            if api_elevation is not None:
                break

        if api_elevation is not None:
            # Check if current elevation is close to API elevation (within 10% tolerance)
            elevation_diff = abs(current_elevation - api_elevation)
            tolerance = max(api_elevation * 0.1, 10)  # 10% tolerance, minimum 10m

            if elevation_diff > tolerance:  # More than 10% difference
                print(f"{Fore.RED}Elevation discrepancy detected:")
                print(f"  Current elevation: {current_elevation}m")
                print(f"  Estimated elevation: {api_elevation:.1f}m")
                print(f"  Difference: {elevation_diff:.1f}m{Style.RESET_ALL}")

                print(f"\n{Fore.CYAN}What would you like to do?{Style.RESET_ALL}")
                print("1. Use estimated elevation value (recommended)")
                print("2. Keep current elevation value")
                print("3. Enter a new elevation value manually")

                while True:
                    choice = input(f"{Fore.CYAN}Enter choice (1-3): {Style.RESET_ALL}").strip()

                    if choice == '1':
                        specs.loc[0, 'Elevation (m)'] = api_elevation
                        elevation_changed = True
                        new_elevation = api_elevation
                        print(f"{Fore.GREEN}Elevation updated to: {api_elevation:.1f}m{Style.RESET_ALL}")
                        break

                    elif choice == '2':
                        print(f"{Fore.GREEN}Keeping current elevation: {current_elevation}m{Style.RESET_ALL}")
                        break

                    elif choice == '3':
                        while True:
                            try:
                                manual_elevation = float(
                                    input(f"{Fore.CYAN}Enter elevation manually: {Style.RESET_ALL}"))
                                specs.loc[0, 'Elevation (m)'] = manual_elevation
                                elevation_changed = True
                                new_elevation = manual_elevation
                                print(f"{Fore.GREEN}Elevation set to: {manual_elevation}m{Style.RESET_ALL}")
                                break
                            except ValueError:
                                print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
                        break
                    else:
                        print(
                            f"{Fore.RED}Invalid choice. Please enter a number between 1 and 3.{Style.RESET_ALL}")
            else:
                print(
                    f"{Fore.GREEN}Elevation looks good: {current_elevation}m (Estimated: {api_elevation:.1f}m){Style.RESET_ALL}")
        else:
            print(
                f"{Fore.YELLOW}Could not estimate elevation for latitude {lat} and longitude {lon}."
                f"Current value: {current_elevation}m{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error verifying elevation: {e}{Style.RESET_ALL}")

    # Save back to original Excel file if elevation was changed, preserving formatting
    if elevation_changed:
        if update_elevation_in_excel('./SystemData/System_Specifications.xlsx', new_elevation):
            print(f"{Fore.GREEN}System_Specifications.xlsx updated with new elevation{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Warning: Could not update System_Specifications.xlsx with preserved formatting{Style.RESET_ALL}")

    return specs


def check_profile_changes(profile_path: str, saved_path: str,
                          is_daynight: bool = False) -> Tuple[pd.DataFrame, ProcessingStage]:
    """
    Check if a profile file has changed since last execution.

    Args:
        profile_path: Path to current profile file
        saved_path: Path to saved profile file in DebugData
        is_daynight: Whether this is a day/night profile (different index)

    Returns:
        Tuple containing:
        - DataFrame with profile data
        - ProcessingStage indicating if recalculation is needed
    """
    try:
        index_col = None if is_daynight else 'Hour of day'
        profile = pd.read_excel(profile_path, index_col=index_col)
    except Exception as e:
        print(f"{Fore.RED}Error reading {os.path.basename(profile_path)}: {e}{Style.RESET_ALL}")
        input(f"{Fore.CYAN}Please fix the file and press Enter to continue...{Style.RESET_ALL}")
        return check_profile_changes(profile_path, saved_path, is_daynight)

    try:
        saved_profile = pd.read_excel(saved_path, index_col=index_col)
        if saved_profile.equals(profile):
            print(f"{Fore.GREEN}No changes to {os.path.basename(profile_path)}{Style.RESET_ALL}")
            return profile, ProcessingStage.NO_CHANGES
        else:
            print(f"{Fore.RED}Changes detected in {os.path.basename(profile_path)}!{Style.RESET_ALL}")
            os.makedirs(os.path.dirname(saved_path), exist_ok=True)
            profile.to_excel(saved_path, index=False)
            return profile, ProcessingStage.STATE_OF_CHARGE

    except Exception:
        print(f"{Fore.YELLOW}No saved {os.path.basename(profile_path)} found. Creating new one...{Style.RESET_ALL}")
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        profile.to_excel(saved_path, index=False)
        return profile, ProcessingStage.STATE_OF_CHARGE


def read_consumption_profile() -> Tuple[pd.DataFrame, ProcessingStage, DataSourceInfo]:
    """
    Read and validate consumption profile data.

    Returns:
        Tuple containing:
        - DataFrame with consumption profile
        - ProcessingStage indicating if recalculation is needed
        - DataSourceInfo with profile type information
    """
    print(f"{Fore.YELLOW}Checking consumption profiles...{Style.RESET_ALL}")

    # Create SystemData directory if it doesn't exist
    system_data_dir = "./SystemData"
    if not os.path.exists(system_data_dir):
        os.makedirs(system_data_dir, exist_ok=True)
        print(f"{Fore.GREEN}Created directory: {system_data_dir}{Style.RESET_ALL}")

    data_source = DataSourceInfo()
    hourly_profile_exists = os.path.exists('./SystemData/Consumption_Profile.xlsx')
    daynight_profile_exists = os.path.exists('./SystemData/Day_Night_Profile.xlsx')

    if not hourly_profile_exists and not daynight_profile_exists:
        print(f"{Fore.RED}No consumption profile files found!")
        print("Please add either:")
        print("- SystemData/Consumption_Profile.xlsx (hourly profile)")
        print(f"- SystemData/Day_Night_Profile.xlsx (day/night profile){Style.RESET_ALL}")
        input(f"{Fore.CYAN}Add files and press Enter to continue...{Style.RESET_ALL}")
        return read_consumption_profile()

    # If both files exist, let user choose
    if hourly_profile_exists and daynight_profile_exists:
        print(f"{Fore.CYAN}Found both profile types. Which would you like to use?{Style.RESET_ALL}")
        print("1. Hourly consumption profile (Consumption_Profile.xlsx)")
        print("2. Day/Night profile (Day_Night_Profile.xlsx)")

        while True:
            choice = input(f"{Fore.CYAN}Enter choice (1/2): {Style.RESET_ALL}").strip()
            if choice == '1':
                data_source.using_day_night = False
                break
            elif choice == '2':
                data_source.using_day_night = True
                break
            print(f"{Fore.RED}Invalid choice. Please enter 1 or 2.{Style.RESET_ALL}")
    else:
        # Use whichever file exists
        data_source.using_day_night = daynight_profile_exists

    # Check the appropriate file
    if data_source.using_day_night:
        profile, stage = check_profile_changes(
            './SystemData/Day_Night_Profile.xlsx',
            './DebugData/Saved_Day_Night_Profile.xlsx',
            is_daynight=True
        )
    else:
        profile, stage = check_profile_changes(
            './SystemData/Consumption_Profile.xlsx',
            './DebugData/Saved_Consumption_Profile.xlsx',
            is_daynight=False
        )

    return profile, stage, data_source


def validate_system_specifications_file() -> pd.DataFrame:
    """
    Validate and read the System_Specifications.xlsx file.
    
    Checks for:
    - File existence and readability
    - Correct column names
    - Valid parameter values and ranges
    - Data types
    
    Returns:
        pd.DataFrame: Validated system specifications
        
    Raises:
        FileNotFoundError: If the specifications file is missing
        ValueError: If validation fails
    """
    file_path = './SystemData/System_Specifications.xlsx'
    
    while True:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"{Fore.RED}System_Specifications.xlsx not found in SystemData folder!")
            input(f"Add the file and press Enter to continue...{Style.RESET_ALL}")
            continue
            
        # Try to read the Excel file
        try:
            specs = pd.read_excel(file_path, index_col=None)
        except Exception as e:
            print(f"{Fore.RED}Cannot read System_Specifications.xlsx: {e}")
            print("This could be due to:")
            print("- Corrupted file format")
            print("- File is not a valid Excel file") 
            print("- File is currently open in another application")
            input(f"Fix the file and press Enter to continue...{Style.RESET_ALL}")
            continue
            
        # Validate column names
        expected_columns = SystemSpecifications.get_all_params()
        missing_columns = [col for col in expected_columns if col not in specs.columns]
        extra_columns = [col for col in specs.columns if col not in expected_columns]
        
        if missing_columns or extra_columns:
            print(f"{Fore.RED}Column validation failed:")
            if missing_columns:
                print(f"Missing columns: {missing_columns}")
            if extra_columns:
                print(f"Unexpected columns: {extra_columns}")
            print(f"Expected columns: {expected_columns}")
            input(f"Fix the column names and press Enter to continue...{Style.RESET_ALL}")
            continue
            
        # Check if we have at least one row of data
        if len(specs) == 0:
            print(f"{Fore.RED}System_Specifications.xlsx contains no data rows!")
            input(f"Add data to the file and press Enter to continue...{Style.RESET_ALL}")
            continue
            
        # Validate data types and ranges
        validation_errors = []
        
        # Validate location parameters
        try:
            lat = float(specs['Lattitude (°)'][0])
            if not (-90 <= lat <= 90):
                validation_errors.append(f"Latitude must be between -90° and 90°, got {lat}°")
        except (ValueError, TypeError):
            validation_errors.append("Latitude must be a valid number")
            
        try:
            lon = float(specs['Longitude (°)'][0])
            if not (-180 <= lon <= 180):
                validation_errors.append(f"Longitude must be between -180° and 180°, got {lon}°")
        except (ValueError, TypeError):
            validation_errors.append("Longitude must be a valid number")
            
        try:
            elevation = float(specs['Elevation (m)'][0])
            if elevation < -500 or elevation > 9000:
                validation_errors.append(f"Elevation {elevation}m seems unrealistic (expected -500m to 9000m)")
        except (ValueError, TypeError):
            validation_errors.append("Elevation must be a valid number")
            
        # Validate calibration parameters
        try:
            pattern_cols = int(specs['Calib vertex short'][0])
            pattern_rows = int(specs['Calib vertex long'][0])
            if pattern_cols < 3 or pattern_rows < 3:
                validation_errors.append(f"Calibration pattern too small: {pattern_cols}x{pattern_rows}, minimum 3x3")
        except (ValueError, TypeError):
            validation_errors.append("Calibration pattern dimensions must be valid integers")
            
        try:
            square_size = float(specs['Calib square size (mm)'][0])
            if square_size <= 0:
                validation_errors.append(f"Calibration square size must be positive, got {square_size}mm")
        except (ValueError, TypeError):
            validation_errors.append("Calibration square size must be a valid positive number")
            
        # Validate angle parameters (0-360 degrees)
        angle_params = ['Image orientation (°)', 'Image inclination (°)', 
                       'Plane orientation (°)', 'Plane inclination (°)']
        for param in angle_params:
            try:
                angle = float(specs[param][0])
                if not (0 <= angle <= 360):
                    validation_errors.append(f"{param} must be between 0° and 360°, got {angle}°")
            except (ValueError, TypeError):
                validation_errors.append(f"{param} must be a valid number")
                
        # Validate time parameters (YYYYMMDD format)
        try:
            start_date = int(specs['Start year'][0])
            end_date = int(specs['End year'][0])
            current_year = pd.Timestamp.now().year
            
            # Validate YYYYMMDD format (8 digits, reasonable year range)
            if not (19000101 <= start_date <= (current_year + 10) * 10000 + 1231):
                validation_errors.append(f"Start date {start_date} should be within reasonable year range")
            if not (19000101 <= end_date <= (current_year + 10) * 10000 + 1231):
                validation_errors.append(f"End date {end_date} should be within reasonable year range")
            if start_date > end_date:
                validation_errors.append(f"Start date {start_date} must be <= end date {end_date}")
                
        except (ValueError, TypeError):
            validation_errors.append("Start year and End year must be valid integers in YYYYMMDD format")
            
        # Validate system parameters (must be positive)
        positive_params = ['Solar panel peak wattage (W)', 'Converter max power (W)',
                          'Batt nominal capacity (Ah)', 'Batt nominal voltage (V)']
        for param in positive_params:
            try:
                value = float(specs[param][0])
                if value <= 0:
                    validation_errors.append(f"{param} must be positive, got {value}")
            except (ValueError, TypeError):
                validation_errors.append(f"{param} must be a valid positive number")
                
        # Validate percentage parameters (0-100%)
        percentage_params = ['Converter efficiency (%)', 'Charge efficiency (%)',
                           'Discharge efficiency (%)', 'Max SOC (%)', 'Min SOC (%)']
        for param in percentage_params:
            try:
                value = float(specs[param][0])
                if not (0 <= value <= 100):
                    validation_errors.append(f"{param} must be between 0% and 100%, got {value}%")
            except (ValueError, TypeError):
                validation_errors.append(f"{param} must be a valid percentage")
                
        # Validate SOC range logic
        try:
            min_soc = float(specs['Min SOC (%)'][0])
            max_soc = float(specs['Max SOC (%)'][0])
            if min_soc >= max_soc:
                validation_errors.append(f"Min SOC ({min_soc}%) must be less than Max SOC ({max_soc}%)")
        except (ValueError, TypeError):
            pass  # Already handled above
            
        # If there are validation errors, report them and ask for fixes
        if validation_errors:
            print(f"{Fore.RED}System specifications validation failed:")
            for error in validation_errors:
                print(f"- {error}")
            input(f"Fix these issues in System_Specifications.xlsx and press Enter to continue...{Style.RESET_ALL}")
            continue
            
        # If we get here, validation passed
        print(f"{Fore.GREEN}System specifications validated successfully.{Style.RESET_ALL}")
        return specs


def read_system_specifications() -> Tuple[pd.DataFrame, ProcessingStage]:
    """
    Read and validate system specifications.

    Returns:
        Tuple containing:
        - DataFrame with system specifications
        - ProcessingStage indicating if recalculation is needed
    """
    print(f"{Fore.YELLOW}Reading system specifications...{Style.RESET_ALL}")

    # Validate and read the specifications file
    specs = validate_system_specifications_file()

    # Keep a copy of original specs for comparison before elevation verification
    original_specs = specs.copy()
    
    # Verify elevation data (may modify specs)
    specs = verify_elevation(specs)

    stage = ProcessingStage.NO_CHANGES

    try:
        saved_specs = pd.read_excel(
            './DebugData/Saved_System_Specifications.xlsx',
            index_col=None,
            dtype={
                'Elevation (m)': float})

        # Check each parameter group for changes using original specs for comparison
        for param in SystemSpecifications.get_all_params():
            original_value = float(original_specs[param][0])
            saved_value = float(saved_specs[param][0])
            
            # Use a small tolerance for floating-point comparison (1e-6 relative tolerance)
            if abs(original_value - saved_value) > max(1e-6 * abs(saved_value), 1e-9):
                print(f"{Fore.YELLOW}Change detected in {param}: "
                      f"{saved_value} -> {original_value}{Style.RESET_ALL}")

                # Update processing stage based on parameter group
                if param in SystemSpecifications.CALIB_PARAMS:
                    stage = min(stage, ProcessingStage.CALIBRATION)
                elif param in SystemSpecifications.LOCATION_PARAMS + SystemSpecifications.IMAGE_PARAMS:
                    stage = min(stage, ProcessingStage.RAW_IRRADIANCE)
                else:
                    stage = min(stage, ProcessingStage.STATE_OF_CHARGE)

        # Save the current specs (potentially updated) if any changes occurred
        if stage != ProcessingStage.NO_CHANGES:
            specs.to_excel('./DebugData/Saved_System_Specifications.xlsx', index=False)

    except Exception:
        print(f"{Fore.YELLOW}No saved specifications found. Creating new one...{Style.RESET_ALL}")
        os.makedirs('./DebugData', exist_ok=True)
        specs.to_excel('./DebugData/Saved_System_Specifications.xlsx', index=False)
        stage = ProcessingStage.CALIBRATION

    return specs, stage


def process_sky_images() -> Tuple[np.ndarray, int, int, bool, str, List[str]]:
    """
    Process and validate sky images.

    Returns:
        Tuple containing:
        - Selected image as numpy array
        - Image height
        - Image width
        - Flag indicating if multiple images should be combined
        - Selected image path
        - List of all images (used when combining)
    """
    print(f"{Fore.YELLOW}Processing sky images...{Style.RESET_ALL}")

    # Create SkyImageOfSite directory if it doesn't exist
    sky_image_dir = "./SkyImageOfSite"
    if not os.path.exists(sky_image_dir):
        os.makedirs(sky_image_dir, exist_ok=True)
        print(f"{Fore.GREEN}Created directory: {sky_image_dir}{Style.RESET_ALL}")

    while True:
        images = [f for f in glob("./SkyImageOfSite/*")
                  if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}]

        if not images:
            input(f"{Fore.RED}No image found in SkyImageOfSite. "
                  f"Add image(s) and press Enter...{Style.RESET_ALL}")
            continue

        selected_image = None
        flag_combination = False

        if len(images) > 1:
            ans = input(f"{Fore.CYAN}Found {len(images)} images. "
                        f"Combine them? (yes/no): {Style.RESET_ALL}").strip().lower()

            if ans in {"yes", "y"}:
                sizes = []
                for img_path in images:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"{Fore.RED}Cannot read {img_path}{Style.RESET_ALL}")
                        continue
                    sizes.append(img.shape[::-1])

                if len(set(sizes)) == 1:
                    flag_combination = True
                    im_width, im_height = sizes[0]
                    # For combination case, we'll use the first image as selected_image
                    # and create a placeholder img array
                    selected_image = images[0]
                    img = np.zeros((im_height, im_width, 3), dtype=np.uint8)  # Placeholder
                    break
                else:
                    print(f"{Fore.RED}Images have different sizes:{Style.RESET_ALL}")
                    for img_path, size in zip(images, sizes):
                        print(f"- {os.path.basename(img_path)}: {size}")
                    input(f"{Fore.RED}Fix image sizes and press Enter...{Style.RESET_ALL}")
                    continue

        # Single image or user chose not to combine
        if not flag_combination:
            if len(images) > 1:
                print(f"{Fore.YELLOW}\nAvailable images:{Style.RESET_ALL}")
                for idx, img_path in enumerate(images, 1):
                    print(f"{idx}. {os.path.basename(img_path)}")

                while True:
                    choice = input(f"{Fore.CYAN}Select image number: {Style.RESET_ALL}").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(images):
                        selected_image = images[int(choice) - 1]
                        break
                    print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")
            else:
                selected_image = images[0]
                print(f"{Fore.YELLOW}Using single image: {os.path.basename(selected_image)}{Style.RESET_ALL}")

            img = cv2.imread(selected_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_height, im_width = img.shape[:2]
            break

    return img, im_height, im_width, flag_combination, selected_image, images


def validate_calibration_images(im_height: int, im_width: int, user_data: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, bool]:
    """
    Validate calibration images.

    Args:
        im_height: Required image height
        im_width: Required image width
        user_data: DataFrame containing system specifications (will be updated if parameters are corrected)

    Returns:
        Tuple containing:
        - List of validated calibration image paths
        - Updated DataFrame with corrected calibration parameters (if any were changed)
        - Boolean indicating if calibration parameters were updated
        
    Raises:
        ValueError: If calibration parameters are invalid
        FileNotFoundError: If calibration images are missing
    """
    # Create CalibrationImages directory if it doesn't exist
    calib_dir = "./CalibrationImages"
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir, exist_ok=True)
        print(f"{Fore.GREEN}Created directory: {calib_dir}{Style.RESET_ALL}")
    
    while True:
        calib_files = glob('./CalibrationImages/*.jpg')

        # Validate minimum number of calibration images
        if len(calib_files) < 8:
            input(f"{Fore.RED}Insufficient calibration images: {len(calib_files)} found, minimum 8 required"
                  f"\nAdd more images and press Enter to continue...{Style.RESET_ALL}")
            continue

        # Check if calibration files exist and are readable
        missing_files = []
        unreadable_files = []
        for file_path in calib_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
            else:
                # Test if image can be read
                test_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if test_img is None:
                    unreadable_files.append(file_path)
        
        if missing_files:
            print(f"{Fore.RED}Calibration images not found: {missing_files}")
            input(f"Fix missing files and press Enter to continue...{Style.RESET_ALL}")
            continue
            
        if unreadable_files:
            print(f"{Fore.RED}Cannot read calibration images: {unreadable_files}")
            input(f"Fix corrupted files and press Enter to continue...{Style.RESET_ALL}")
            continue

        # Get calibration parameters from user_data (already validated in validate_system_specifications_file)
        pattern_cols = int(user_data['Calib vertex short'][0])
        pattern_rows = int(user_data['Calib vertex long'][0])
        square_size = float(user_data['Calib square size (mm)'][0])
        
        print(f"{Fore.CYAN}Using calibration parameters: {pattern_cols}x{pattern_rows}, square size: {square_size}mm{Style.RESET_ALL}")

        # Validate image sizes
        size_mismatch = False
        mismatched_sizes = set()
        for fname in calib_files:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img.shape != (im_height, im_width):
                mismatched_sizes.add((img.shape))
                size_mismatch = True

        if size_mismatch:
            print(f"{Fore.RED}Size mismatch in calibration images:")
            print(f"Expected size (sky image): {im_width}x{im_height}")
            print(f"Got: {', '.join([f'{size[1]}x{size[0]}' for size in mismatched_sizes])}")
            input(f"{Fore.RED}Fix image sizes and press Enter...{Style.RESET_ALL}")
            continue

        print(f"{Fore.GREEN}Calibration images validated successfully: {len(calib_files)} images{Style.RESET_ALL}")
        return calib_files, user_data


def update_image_status(images: List[str], calib_files: List[str]) -> ProcessingStage:
    """
    Update and check image status file.

    Args:
        images: List of sky image paths
        calib_files: List of calibration image paths

    Returns:
        ProcessingStage indicating if recalculation is needed
    """
    # Convert stat results to lists and create deep copies to modify
    sky_meta = [list(os.stat(f)) for f in images]
    calib_meta = [list(os.stat(f)) for f in calib_files]

    # Ignore access time (st_atime is at index 7)
    for meta in sky_meta + calib_meta:
        meta[7] = 0

    stage = ProcessingStage.NO_CHANGES

    # Get the current status data if it exists
    if os.path.exists('status.yml'):
        try:
            with open('status.yml', 'r') as f:
                data = yaml.safe_load(f) or {}

            # Check if metadata exists and compare
            if 'sky_image_metadata' in data and data['sky_image_metadata'] != sky_meta:
                print(f"{Fore.RED}Sky image changes detected!{Style.RESET_ALL}")
                stage = ProcessingStage.RAW_IRRADIANCE

            if 'calib_images_metadata' in data and data['calib_images_metadata'] != calib_meta:
                print(f"{Fore.RED}Calibration image changes detected!{Style.RESET_ALL}")
                stage = ProcessingStage.CALIBRATION

        except Exception as e:
            print(f"{Fore.YELLOW}Error reading image metadata: {e}. Will recreate.{Style.RESET_ALL}")
            stage = ProcessingStage.CALIBRATION
    else:
        # If status file doesn't exist, we need to recalculate from the beginning
        print(f"{Fore.YELLOW}No image metadata found. Need to recalculate from scratch.{Style.RESET_ALL}")
        stage = ProcessingStage.CALIBRATION

    # Update the status file with new metadata (preserve other data)
    try:
        # Try to read existing data first
        if os.path.exists('status.yml'):
            with open('status.yml', 'r') as f:
                status_data = yaml.safe_load(f) or {}
        else:
            status_data = {}

        # Update image metadata
        status_data['sky_image_metadata'] = sky_meta
        status_data['calib_images_metadata'] = calib_meta

        # Write back to file
        with open('status.yml', 'w') as f:
            yaml.dump(status_data, f)

    except Exception as e:
        print(f"{Fore.RED}Could not update image status: {e}")
        print(f"Recreating status.yml file...{Style.RESET_ALL}")
        try:
            # Since we recreated the status file, we need to recalculate everything from calibration
            stage = ProcessingStage.CALIBRATION
            # Recreate the status file with just the image metadata
            status_data = {
                'sky_image_metadata': sky_meta,
                'calib_images_metadata': calib_meta
            }
            with open('status.yml', 'w') as f:
                yaml.dump(status_data, f)
            print(f"{Fore.GREEN}Status file recreated successfully.{Style.RESET_ALL}")
        except Exception as recreate_error:
            print(f"{Fore.RED}Critical Error: Could not recreate status.yml: {recreate_error}")
            print(f"This indicates a serious system issue (disk full, permissions, etc.)")
            print(f"Please check your system and try again.{Style.RESET_ALL}")
            raise RuntimeError(f"Unable to create or write to status.yml file: {recreate_error}") from recreate_error

    return stage


def read_user_data() -> Tuple[pd.DataFrame, List[str], ProcessingStage,
                              int, int, bool, np.ndarray, DataSourceInfo]:
    """
    Read and validate all user data and determine processing stage.

    Returns:
        Tuple containing:
        - DataFrame with system specifications
        - List of calibration file paths
        - ProcessingStage indicating what needs to be recalculated
        - Image height
        - Image width
        - Flag indicating if multiple images should be combined
        - Selected image as numpy array
        - DataSourceInfo with profile and irradiance source information
    """
    # Initialize processing stage to NO_CHANGES
    stage = ProcessingStage.NO_CHANGES

    # Initialize data source with defaults
    data_source = DataSourceInfo()

    # Try to load previous settings from status file if it exists
    if os.path.exists('status.yml'):
        try:
            with open('status.yml', 'r') as f:
                data = yaml.safe_load(f) or {}
                if 'data_source' in data:
                    # Update data source with stored preferences
                    data_source.using_nasa = data['data_source'].get('using_nasa', False)
                    data_source.using_day_night = data['data_source'].get('using_day_night', False)
                    data_source.last_irradiance_file = data['data_source'].get(
                        'last_irradiance_file', "")
        except Exception as e:
            print(f"{Fore.RED}Error loading status file: {e}. Using defaults.{Style.RESET_ALL}")

    # Step 1: Read consumption profile
    _, consumption_stage, profile_info = read_consumption_profile()
    stage = ProcessingStage.get_min_stage(stage, consumption_stage)
    data_source.using_day_night = profile_info.using_day_night

    # Step 2: Read system specifications
    specs, specs_stage = read_system_specifications()
    stage = ProcessingStage.get_min_stage(stage, specs_stage)

    # Step 3: Process sky images
    img, im_height, im_width, flag_combination, selected_image, all_images = process_sky_images()

    # Step 4: Validate calibration images and update user_data with corrected parameters
    calib_files, specs = validate_calibration_images(im_height, im_width, specs)

    # Step 5: Check for image changes
    # Use all images if combining, otherwise just the selected image
    images_to_check = all_images if flag_combination else [selected_image]
    image_stage = update_image_status(images_to_check, calib_files)
    stage = ProcessingStage.get_min_stage(stage, image_stage)

    # Update data source info in the status file
    try:
        # Try to read existing data first (create if doesn't exist)
        if os.path.exists('status.yml'):
            with open('status.yml', 'r') as f:
                status_data = yaml.safe_load(f) or {}
        else:
            status_data = {}

        # Update data source info
        status_data['data_source'] = {
            'using_nasa': data_source.using_nasa,
            'using_day_night': data_source.using_day_night,
            'last_irradiance_file': data_source.last_irradiance_file
        }

        # Write back to file
        with open('status.yml', 'w') as f:
            yaml.dump(status_data, f)

    except Exception as e:
        print(f"{Fore.RED}Error updating status file: {e}{Style.RESET_ALL}")

    print(f"{Fore.GREEN}Data validation complete. Stage: {stage.name}")
    print(f"Using {'day/night' if data_source.using_day_night else 'hourly'} profile")
    print(f"Using {'NASA POWER' if data_source.using_nasa else 'user-provided'} irradiance data{Style.RESET_ALL}")

    return specs, calib_files, stage, im_height, im_width, flag_combination, img, data_source
