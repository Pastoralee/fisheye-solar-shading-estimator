import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style
from config import PATHS, DataSourceInfo
from validation import (
    get_user_choice,
    prompt_for_file,
    validate_file_exists,
    validate_images,
    validate_system_specs,
)
from elevation_api import verify_and_update_elevation
try:
    from pvlib import irradiance, solarposition
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False


def load_system_specs() -> pd.DataFrame:
    """Load and validate system specifications from Excel file.
    
    Continuously prompts the user to provide a valid System_Specifications.xlsx file
    until one is found and successfully validated. Includes elevation verification
    using external API data.
    
    Returns:
        pd.DataFrame: Validated system specifications containing location parameters,
                     system parameters, and configuration settings
                     
    Raises:
        Exception: If file reading or validation fails repeatedly
    """
    while True:
        if not validate_file_exists(PATHS["system_specs"], "System specifications"):
            prompt_for_file("Add System_Specifications.xlsx to SystemData folder.")
            continue

        try:
            specs = pd.read_excel(PATHS["system_specs"])
            if validate_system_specs(specs):
                print(f"{Fore.GREEN}System specifications loaded successfully{Style.RESET_ALL}")
                
                # Verify elevation using API data
                specs = verify_and_update_elevation(specs, PATHS["system_specs"])
                
                return specs
        except Exception as e:
            print(f"{Fore.RED}Error reading system specs: {e}{Style.RESET_ALL}")

        prompt_for_file("Fix the system specifications file.")


def load_consumption_profile() -> Tuple[pd.DataFrame, DataSourceInfo]:
    """Load consumption profile from Excel file (hourly or day/night patterns).
    
    Checks for both hourly consumption profiles and day/night profiles, 
    prompting the user to choose if both are available.
    
    Returns:
        Tuple containing:
            - pd.DataFrame: Loaded consumption profile data
            - DataSourceInfo: Information about which data source was used
            
    Raises:
        Exception: If profile files are missing or corrupted
    """
    data_source = DataSourceInfo()

    # Create SystemData directory if needed
    os.makedirs(PATHS["system_data"], exist_ok=True)

    hourly_exists = os.path.exists(PATHS["consumption_profile"])
    daynight_exists = os.path.exists(PATHS["day_night_profile"])

    if not hourly_exists and not daynight_exists:
        print(f"{Fore.RED}No consumption profile found!{Style.RESET_ALL}")
        prompt_for_file(
            "Add either Consumption_Profile.xlsx or Day_Night_Profile.xlsx to SystemData."
        )
        return load_consumption_profile()

    # Choose profile type
    if hourly_exists and daynight_exists:
        print(f"{Fore.CYAN}Both profile types found. Choose:{Style.RESET_ALL}")
        print("1. Hourly consumption profile")
        print("2. Day/Night profile")
        choice = get_user_choice("Enter choice (1/2): ", ["1", "2"])
        data_source.using_day_night = choice == "2"
    else:
        data_source.using_day_night = daynight_exists

    # Load the appropriate profile
    try:
        if data_source.using_day_night:
            profile = pd.read_excel(PATHS["day_night_profile"])
            print(f"{Fore.GREEN}Day/Night profile loaded{Style.RESET_ALL}")
        else:
            profile = pd.read_excel(PATHS["consumption_profile"], index_col="Hour of day")
            print(f"{Fore.GREEN}Hourly profile loaded{Style.RESET_ALL}")
        return profile, data_source
    except Exception as e:
        print(f"{Fore.RED}Error reading consumption profile: {e}{Style.RESET_ALL}")
        prompt_for_file("Fix the consumption profile file.")
        return load_consumption_profile()


def load_sky_images() -> Tuple[np.ndarray, bool, List[str]]:
    """Load and process sky images for shading analysis.
    
    Validates sky images in the designated folder and handles single/multiple
    image scenarios. For multiple images, allows user choice between combining
    them or selecting a single image for processing.
    
    Returns:
        Tuple containing:
            - np.ndarray: Loaded sky image data (RGB format), or None for batch processing
            - bool: Flag indicating whether multiple images should be combined (batch processing)
            - List[str]: List of image file paths to process
            
    Note:
        Recursively calls itself if no valid images are found until user provides them.
    """
    images = validate_images(PATHS["sky_images"])
    if not images:
        prompt_for_file("Add sky images to SkyImageOfSite folder.")
        return load_sky_images()

    # Handle multiple images
    if len(images) > 1:
        choice = get_user_choice(
            f"Found {len(images)} images. Combine them? (y/n): ", ["y", "n", "yes", "no"]
        )
        flag_combination = choice.lower() in ["y", "yes"]
    else:
        flag_combination = False

    if flag_combination:
        print(f"{Fore.YELLOW}Using batch processing for {len(images)} images{Style.RESET_ALL}")
        return None, True, images
    else:
        # Use single image
        if len(images) > 1:
            print(f"{Fore.CYAN}Available images:{Style.RESET_ALL}")
            for i, img_path in enumerate(images):
                print(f"{i+1}. {os.path.basename(img_path)}")

            # Create valid choices list
            valid_choices = [str(i + 1) for i in range(len(images))]
            choice = get_user_choice("Select image number: ", valid_choices)
            selected = images[int(choice) - 1]
        else:
            selected = images[0]

        # Load the selected image
        img = cv2.imread(selected)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"{Fore.GREEN}Loaded sky image: {os.path.basename(selected)}{Style.RESET_ALL}")
        return img, False, [selected]


def load_calibration_images() -> List[str]:
    """Load calibration images for camera calibration process.
    
    Validates that at least 8 JPEG calibration images are present in the
    CalibrationImages folder, as required for accurate camera calibration.
    
    Returns:
        List[str]: List of file paths to valid calibration images
        
    Note:
        Recursively calls itself if insufficient images are found until
        user provides the required minimum number of calibration images.
    """
    images = validate_images(PATHS["calibration_images"], min_count=8, extensions=[".jpg"])
    if not images:
        prompt_for_file("Add at least 8 calibration images to CalibrationImages folder.")
        return load_calibration_images()

    print(f"{Fore.GREEN}Found {len(images)} calibration images{Style.RESET_ALL}")
    return images


def load_status_file() -> dict:
    """Load status file or create empty dictionary if file doesn't exist.
    
    Attempts to load the YAML status file that tracks processing pipeline state.
    If the file doesn't exist or is corrupted, returns an empty dictionary.
    
    Returns:
        dict: Status data loaded from YAML file, or empty dict if loading fails
    """
    if os.path.exists(PATHS["status_file"]):
        try:
            with open(PATHS["status_file"], "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def save_status_file(status_data: dict) -> None:
    """Save status data to YAML file.
    
    Persists the current pipeline processing status to a YAML file.
    Handles exceptions gracefully by printing warnings if save fails.
    
    Args:
        status_data (dict): Dictionary containing pipeline status information
                           to be saved to the status file
    """
    try:
        with open(PATHS["status_file"], "w") as f:
            yaml.dump(status_data, f, default_flow_style=False)
    except Exception as e:
        print(f"{Fore.YELLOW}Warning: Could not save status file: {e}{Style.RESET_ALL}")


def load_custom_irradiance_data(
    time_array: pd.DatetimeIndex, lat: float, lon: float
) -> Tuple[Optional[pd.DatetimeIndex], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Load irradiance data from custom CSV/Excel file in SystemData folder.
    
    Allows users to select and load irradiance data from CSV or Excel files,
    with support for GHI decomposition using the Erbs model.
    
    Args:
        time_array: Reference time array for solar calculations
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        Tuple containing:
            - pd.DatetimeIndex: Datetime column from the loaded file
            - np.ndarray: Direct irradiance values (DNI or decomposed)
            - np.ndarray: Diffuse irradiance values (DHI or decomposed)
            - str: Irradiance type ('normal' for DNI, 'horizontal' for GHI)
            
        Returns (None, None, None, None) if loading fails or no files available.
    """

    def get_available_files() -> List[str]:
        """Get list of CSV and Excel files in SystemData folder.
        
        Excludes system specification and consumption profile files that are
        used for configuration rather than irradiance data.
        
        Returns:
            List[str]: List of available file names with supported extensions,
                      excluding system configuration files
        """
        system_data_path = Path(PATHS["system_data"])
        files = []

        # Files to exclude from the list (system configuration files from PATHS)
        excluded_files = {
            os.path.basename(PATHS["system_specs"]),
            os.path.basename(PATHS["consumption_profile"]), 
            os.path.basename(PATHS["day_night_profile"])
        }

        if system_data_path.exists():
            for ext in [".csv", ".xlsx", ".xls"]:
                found_files = system_data_path.glob(f"*{ext}")
                for file_path in found_files:
                    if file_path.name not in excluded_files:
                        files.append(file_path)

        return [f.name for f in files]

    def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
        """Detect datetime column by name patterns or data types.
        
        Tries to identify a datetime column first by checking data types,
        then by checking column names for common datetime indicators.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze for datetime columns
        
        Returns:
            Optional[str]: Name of the detected datetime column, or None if no
                          datetime column is found
        """
        # First check by data type
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        # Then check by name patterns
        for col in df.columns:
            if any(word in col.lower() for word in ["date", "time", "timestamp", "datetime", "dt"]):
                return col

        return None

    def apply_erbs_model(ghi: np.ndarray, time_array: pd.DatetimeIndex, lat: float, lon: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """Apply Erbs model to decompose GHI into direct and diffuse components.
        
        Uses the Erbs correlation model to separate Global Horizontal Irradiance (GHI)
        into Direct Normal Irradiance (DNI) and Diffuse Horizontal Irradiance (DHI).
        
        Args:
            ghi (np.ndarray): Global Horizontal Irradiance values in W/m²
            time_array (pd.DatetimeIndex): Timestamp array corresponding to the GHI data
            lat (float): Latitude in decimal degrees
            lon (float): Longitude in decimal degrees
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]: 
                - Direct Normal Irradiance (DNI) values, or None if pvlib unavailable
                - Diffuse Horizontal Irradiance (DHI) values, or None if pvlib unavailable
                - String "normal" indicating DNI type, or None if pvlib unavailable
        
        Note:
            Requires pvlib package for solar position calculations and Erbs decomposition.
        """
        if not PVLIB_AVAILABLE:
            print(f"{Fore.RED}pvlib package not available for Erbs decomposition{Style.RESET_ALL}")
            return None, None, None

        # Calculate solar position
        solar_pos = solarposition.get_solarposition(time_array, lat, lon)

        # Apply Erbs decomposition
        erbs_result = irradiance.erbs(ghi, solar_pos["zenith"], time_array)

        return erbs_result["dni"].values, erbs_result["dhi"].values, "normal"

    # List available files
    available_files = get_available_files()

    if not available_files:
        print(f"{Fore.RED}No CSV or Excel files found in SystemData folder{Style.RESET_ALL}")
        return None, None, None, None

    print(f"{Fore.CYAN}Available irradiance files in SystemData:{Style.RESET_ALL}")
    for i, filename in enumerate(available_files, 1):
        print(f"{i:2d}. {filename}")

    # Get user file selection
    valid_choices = [str(i) for i in range(1, len(available_files) + 1)]
    choice = get_user_choice("Select file number: ", valid_choices)
    selected_file = available_files[int(choice) - 1]

    file_path = os.path.join(PATHS["system_data"], selected_file)

    try:
        # Load file (CSV or Excel)
        if selected_file.lower().endswith(".csv"):
            df = pd.read_csv(file_path, sep=None, engine="python")
        else:  # Excel file
            df = pd.read_excel(file_path)

        if df.empty:
            print(f"{Fore.RED}File is empty{Style.RESET_ALL}")
            return None, None, None, None

        print(f"{Fore.GREEN}File loaded successfully! Shape: {df.shape}{Style.RESET_ALL}")

        # Detect datetime column
        date_col = detect_datetime_column(df)

        if date_col is None:
            print(f"{Fore.YELLOW}Available columns:{Style.RESET_ALL}")
            for i, col in enumerate(df.columns, 1):
                print(f"{i:2d}. {col}")

            valid_choices = [str(i) for i in range(1, len(df.columns) + 1)]
            choice = get_user_choice("Select datetime column number: ", valid_choices)
            date_col = df.columns[int(choice) - 1]

        print(f"{Fore.GREEN}Using '{date_col}' as datetime column.{Style.RESET_ALL}")

        # Convert datetime and set as index
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

        if df.empty:
            print(f"{Fore.RED}No valid datetime data found{Style.RESET_ALL}")
            return None, None, None, None

        # Get remaining columns (excluding datetime)
        remaining_cols = [col for col in df.columns]

        if len(remaining_cols) == 1:
            # Single column - assume it's GHI
            ghi_col = remaining_cols[0]
            print(
                f"{Fore.YELLOW}Single data column detected: Using '{ghi_col}' as GHI{Style.RESET_ALL}"
            )
            print(
                f"{Fore.YELLOW}Will apply Erbs model to decompose into direct and diffuse{Style.RESET_ALL}"
            )

            ghi_values = df[ghi_col].values
            direct_irr, diffuse_irr, irr_type = apply_erbs_model(ghi_values, df.index, lat, lon)

            return df.index, direct_irr, diffuse_irr, irr_type

        else:
            # Multiple columns - let user choose approach
            print(f"{Fore.CYAN}Data processing options:{Style.RESET_ALL}")
            print("1. Use GHI (Global Horizontal Irradiance) - will apply Erbs model")
            print("2. Use DNI + DHI (Direct Normal + Diffuse Horizontal)")
            print("3. Use BHI + DHI (Beam Horizontal + Diffuse Horizontal)")

            choice = get_user_choice("Enter choice (1/2/3): ", ["1", "2", "3"])

            if choice == "1":
                # GHI selection
                print(f"{Fore.YELLOW}Available columns for GHI:{Style.RESET_ALL}")
                for i, col in enumerate(remaining_cols, 1):
                    print(f"{i:2d}. {col}")

                valid_choices = [str(i) for i in range(1, len(remaining_cols) + 1)]
                ghi_choice = get_user_choice("Select GHI column number: ", valid_choices)
                ghi_col = remaining_cols[int(ghi_choice) - 1]

                ghi_values = df[ghi_col].values
                direct_irr, diffuse_irr, irr_type = apply_erbs_model(ghi_values, df.index, lat, lon)

                return df.index, direct_irr, diffuse_irr, irr_type

            else:
                # Direct + Diffuse selection
                irr_type = "normal" if choice == "2" else "horizontal"
                direct_name = "DNI" if choice == "2" else "BHI"

                print(f"{Fore.YELLOW}Available columns:{Style.RESET_ALL}")
                for i, col in enumerate(remaining_cols, 1):
                    print(f"{i:2d}. {col}")

                # Select direct irradiance column
                valid_choices = [str(i) for i in range(1, len(remaining_cols) + 1)]
                direct_choice = get_user_choice(
                    f"Select {direct_name} column number: ", valid_choices
                )
                direct_col = remaining_cols[int(direct_choice) - 1]

                # Select diffuse irradiance column - show all columns but exclude already selected
                print(
                    f"\n{Fore.YELLOW}Available columns for DHI (excluding {direct_col}):{Style.RESET_ALL}"
                )
                available_diffuse_cols = [
                    (i, col) for i, col in enumerate(remaining_cols, 1) if i != int(direct_choice)
                ]

                for idx, (orig_num, col) in enumerate(available_diffuse_cols, 1):
                    print(f"{idx:2d}. {col}")

                valid_diffuse_choices = [str(i) for i in range(1, len(available_diffuse_cols) + 1)]
                diffuse_choice = get_user_choice(
                    "Select DHI column number: ", valid_diffuse_choices
                )
                diffuse_col = available_diffuse_cols[int(diffuse_choice) - 1][1]  # Get the column name

                direct_irr = df[direct_col].values
                diffuse_irr = df[diffuse_col].values

                return df.index, direct_irr, diffuse_irr, irr_type

    except Exception as e:
        print(f"{Fore.RED}Error loading file: {e}{Style.RESET_ALL}")
        return None, None, None, None
