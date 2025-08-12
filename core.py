import os
import sys
import traceback
import logging
import re
from enum import IntEnum
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from colorama import Fore, Style
from pvlib.solarposition import get_solarposition
from pvlib.irradiance import erbs
from read_user_data import read_user_data, DataSourceInfo
from calibrate_camera import calibrate_camera
from sunpath_from_astropy import sunpath_from_astropy
from retrieve_NASA_POWER_irradiance import retrieve_NASA_POWER_irradiance
from compute_direct_shading_factor import (
    compute_direct_shading_factor_generic
)
from compute_diffuse_shading_factor import compute_diffuse_shading_factor
from import_camera_intrinsic_function import import_camera_intrinsic_function
from inference import inference, batch_disk_mask_inference
from state_of_charge_estimation import (
    state_of_charge_estimation, 
    state_of_charge_estimation_day_night
)
from dataclasses import dataclass
import yaml


# Logger will be configured in initialize_error_logging()
logger = logging.getLogger(__name__)


class ProcessingStage(IntEnum):
    """Enumeration of processing stages to determine what needs to be recalculated."""
    CALIBRATION = 0          # Calibration needs to be redone
    RAW_IRRADIANCE = 1       # Raw irradiance needs to be recalculated
    SHADING = 2             # Shading needs to be recalculated
    STATE_OF_CHARGE = 3      # State of charge estimation needs to be recalculated
    NO_CHANGES = 4          # No recalculation needed


@dataclass
class ModelConfig:
    """Configuration for sky segmentation model."""
    model_name: str
    use_lgbm: bool
    resize_target: Tuple[int, int]


@dataclass
class IrradianceData:
    """Container for irradiance calculation results."""
    direct: np.ndarray
    diffuse: np.ndarray
    solar_azimuth: np.ndarray
    solar_zenith: np.ndarray
    time_index: pd.DatetimeIndex


def strip_color_codes(text: str) -> str:
    """
    Remove ANSI color codes from text for clean log file output.
    
    This function removes colorama formatting codes to ensure log files
    contain clean, readable text without terminal escape sequences.
    
    Args:
        text: Input text that may contain ANSI color codes
        
    Returns:
        str: Text with all ANSI escape sequences removed
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def log_detailed_error(e: Exception, context: str = "", function_name: str = "") -> str:
    """
    Log detailed error information including stack trace, file, and line number.
    
    Args:
        e: The exception that occurred
        context: Additional context about what was being done when error occurred
        function_name: Name of the function where error occurred
        
    Returns:
        str: Formatted error message for user display
    """
    # Get the current exception info
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    # Get the traceback details
    tb_list = traceback.extract_tb(exc_traceback)
    
    # Find the last frame in our code (not in libraries)
    our_frame = None
    for frame in reversed(tb_list):
        if 'solar_estimation' in frame.filename or 'main.py' in frame.filename:
            our_frame = frame
            break
    
    if our_frame is None and tb_list:
        our_frame = tb_list[-1]  # Fallback to last frame
    
    # Format detailed error information
    error_details = []
    error_details.append(f"{Fore.RED}{'='*60}")
    error_details.append(f"ERROR OCCURRED: {type(e).__name__}")
    error_details.append(f"{'='*60}{Style.RESET_ALL}")
    
    if function_name:
        error_details.append(f"{Fore.YELLOW}Function: {function_name}{Style.RESET_ALL}")
    
    if context:
        error_details.append(f"{Fore.YELLOW}Context: {context}{Style.RESET_ALL}")
    
    if our_frame:
        filename = os.path.basename(our_frame.filename)
        error_details.append(f"{Fore.CYAN}File: {filename}{Style.RESET_ALL}")
        error_details.append(f"{Fore.CYAN}Line: {our_frame.lineno}{Style.RESET_ALL}")
        error_details.append(f"{Fore.CYAN}Code: {our_frame.line}{Style.RESET_ALL}")
    
    error_details.append(f"{Fore.RED}Error Message: {str(e)}{Style.RESET_ALL}")
    
    # Add full traceback for debugging
    error_details.append(f"\n{Fore.MAGENTA}Full Traceback:{Style.RESET_ALL}")
    for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
        error_details.append(f"{Fore.MAGENTA}{line.rstrip()}{Style.RESET_ALL}")
    
    error_message = "\n".join(error_details)
    
    # Log to file with color codes stripped
    clean_error_msg = strip_color_codes(str(e))
    logger.error(f"Error in {function_name}: {context} - {clean_error_msg}", exc_info=True)
    
    # Print to console with colors
    print(error_message)
    
    return error_message


def handle_calibration_error(e: Exception) -> None:
    """
    Handle camera calibration specific errors with detailed guidance.
    
    Args:
        e: The exception that occurred during calibration
    """
    context = "Camera calibration process"
    log_detailed_error(e, context, "process_calibration")
    
    print(f"\n{Fore.RED}CALIBRATION TROUBLESHOOTING GUIDE:{Style.RESET_ALL}")
    print("1. Check that CalibrationImages/ folder contains at least 8 images")
    print("2. Verify checkerboard pattern is clearly visible in images")
    print("3. Ensure checkerboard parameters match your printed pattern")
    print("4. Check image quality (not blurry, good lighting)")
    print("5. Verify OmniCalib dependencies are properly installed")


def handle_irradiance_error(e: Exception, data_source: DataSourceInfo) -> None:
    """
    Handle irradiance data processing errors with specific guidance.
    
    Args:
        e: The exception that occurred during irradiance processing
        data_source: Information about the data source being used
    """
    context = f"Processing {'NASA POWER' if data_source.using_nasa else 'CSV'} irradiance data"
    log_detailed_error(e, context, "process_irradiance_data")
    
    print(f"\n{Fore.RED}IRRADIANCE DATA TROUBLESHOOTING GUIDE:{Style.RESET_ALL}")
    if data_source.using_nasa:
        print("1. Check internet connection for NASA POWER API access")
        print("2. Verify coordinates are within NASA POWER coverage")
        print("3. Check if API is temporarily unavailable")
        print("4. Verify date range is valid (not too far in the future)")
    else:
        print("1. Check CSV file exists in SystemData/ folder")
        print("2. Verify CSV format has datetime and irradiance columns")
        print("3. Check for missing or invalid data values")
        print("4. Ensure datetime format is parseable")


def handle_shading_error(e: Exception) -> None:
    """
    Handle shading calculation errors with detailed guidance.
    
    Args:
        e: The exception that occurred during shading calculation
    """
    context = "Shading factor calculation"
    log_detailed_error(e, context, "process_shading")
    
    print(f"\n{Fore.RED}SHADING CALCULATION TROUBLESHOOTING GUIDE:{Style.RESET_ALL}")
    print("1. Check that sky images exist in SkyImageOfSite/ folder")
    print("2. Verify camera calibration was completed successfully")
    print("3. Check deep learning model files are available")
    print("4. Ensure system specifications are correctly formatted")
    print("5. Verify image dimensions match calibration parameters")


def handle_battery_error(e: Exception) -> None:
    """
    Handle battery modeling errors with detailed guidance.

    Args:
        e: The exception that occurred during battery modeling
    """
    context = "Battery state of charge calculation"
    log_detailed_error(e, context, "process_state_of_charge")
    
    print(f"\n{Fore.RED}BATTERY MODELING TROUBLESHOOTING GUIDE:{Style.RESET_ALL}")
    print("1. Check consumption profile exists in SystemData/")
    print("2. Verify battery parameters are realistic values")
    print("3. Check that irradiance data is available")
    print("4. Ensure output directory is writable")
    print("5. Verify time arrays are properly aligned")


def handle_data_loading_error(e: Exception) -> None:
    """
    Handle data loading errors with specific guidance.
    
    Args:
        e: The exception that occurred during data loading
    """
    log_detailed_error(e, "Loading and validating user data", "main")
    print(f"\n{Fore.RED}DATA LOADING TROUBLESHOOTING:{Style.RESET_ALL}")
    print("1. Check that SystemData/System_Specifications.xlsx exists and is readable")
    print("2. Verify SkyImageOfSite/ folder contains sky images")
    print("3. Check CalibrationImages/ folder for calibration images")
    print("4. Ensure consumption profile exists in SystemData/")
    print("5. Check file permissions and Excel file is not open")


def handle_time_solar_error(e: Exception) -> None:
    """
    Handle time range and solar position calculation errors.
    
    Args:
        e: The exception that occurred during time/solar processing
    """
    log_detailed_error(e, "Parsing time range and calculating solar position", "main")
    print(f"\n{Fore.YELLOW}TIME/SOLAR CALCULATION TROUBLESHOOTING:{Style.RESET_ALL}")
    print("1. Check date format in System_Specifications.xlsx (YYYYMMDD)")
    print("2. Verify coordinates are valid (lat: -90 to 90, lon: -180 to 180)")
    print("3. Check elevation value is reasonable")
    print("4. Ensure astropy library is properly installed")


def handle_saved_irradiance_error(e: Exception) -> ProcessingStage:
    """
    Handle errors when loading saved irradiance data.
    
    Manages situations where previously calculated irradiance data cannot
    be loaded, automatically falling back to recalculating from raw data.
    
    Args:
        e: The exception that occurred during irradiance data loading
        
    Returns:
        ProcessingStage.RAW_IRRADIANCE to trigger recalculation
    """
    log_detailed_error(e, "Loading saved irradiance data", "main")
    print(f"{Fore.RED}Error loading saved irradiance data.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Will recalculate irradiance data from scratch.{Style.RESET_ALL}")
    return ProcessingStage.RAW_IRRADIANCE


def handle_shading_results_error(e: Exception) -> ProcessingStage:
    """
    Handle errors when loading previous shading results.
    
    Manages situations where previously calculated shading data cannot
    be loaded, resetting the processing pipeline to recalculate shading
    factors from the sky images.
    
    Args:
        e: The exception that occurred during shading results loading
        
    Returns:
        ProcessingStage.SHADING to trigger shading recalculation
    """
    log_detailed_error(e, "Loading previous shading results", "main")
    print(f"{Fore.RED}Error loading shaded irradiance data.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}You need to run the shading calculation first.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Resetting to shading stage...{Style.RESET_ALL}")
    return ProcessingStage.SHADING


def show_welcome_message() -> None:
    """
    Display welcome message and prerequisites.
    """
    print(f"{Fore.CYAN}=== Solar Estimation System ==={Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Please ensure you have:\n"
          "1. Calibration images in CalibrationImages/\n"
          "2. Sky photo(s) in SkyImageOfSite/\n"
          "3. Consumption profile (hourly or day/night) in SystemData/\n"
          f"4. System specifications in SystemData/System_Specifications.xlsx{Style.RESET_ALL}\n")


def select_model_configuration() -> ModelConfig:
    """
    Let user select model configuration for sky image processing.

    Returns:
        ModelConfig: Selected model configuration containing model name,
                    whether to use LGBM, and resize target dimensions
    """
    print(f"{Fore.CYAN}Please select a model configuration:")
    print(f"From fastest and lowest performance, to slowest but best performance:{Style.RESET_ALL}")
    print("1. 512x512 EfficientNet-b5")
    print("2. 512x512 EfficientNet-b7")
    print("3. Base EfficientNet-b5")
    print("4. Base EfficientNet-b7 (recommended)")
    print("5. Base EfficientNet-b5 + LGBM")
    print("6. Base EfficientNet-b7 + LGBM")

    configs = {
        "1": ModelConfig("efficientnet-b5", False, (512, 512)),
        "2": ModelConfig("efficientnet-b7", False, (512, 512)),
        "3": ModelConfig("efficientnet-b5", False, (1024, 1024)),
        "4": ModelConfig("efficientnet-b7", False, (1024, 1024)),
        "5": ModelConfig("efficientnet-b5", True, (1024, 1024)),
        "6": ModelConfig("efficientnet-b7", True, (1024, 1024))
    }

    while True:
        choice = input(f"{Fore.CYAN}Enter choice (1-6): {Style.RESET_ALL}").strip()
        if choice in configs:
            return configs[choice]
        print(f"{Fore.RED}Invalid choice. Please enter a number between 1 and 6.{Style.RESET_ALL}")


def process_sky_image(model_config: ModelConfig, flag_combination: bool,
                      img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Process sky image(s) using the selected model configuration.

    This function uses either batch processing for multiple images or single image
    processing based on the flag_combination parameter. It applies the selected
    deep learning model (with optional LGBM) to segment the sky in the image.

    Args:
        model_config: Model configuration settings including model name,
                     LGBM usage flag, and resize dimensions
        flag_combination: Whether to process multiple images (True) or a single image (False)
        img: The input image as a numpy array (required for single image processing)

    Returns:
        Optional[np.ndarray]: Binary mask where 1 indicates sky and 0 indicates non-sky regions,
                             or None if processing fails
    """
    if flag_combination:
        return batch_disk_mask_inference(
            model_name=model_config.model_name,
            use_lgbm=model_config.use_lgbm,
            resize_target=model_config.resize_target
        )
    else:
        return inference(
            img,
            model_name=model_config.model_name,
            use_lgbm=model_config.use_lgbm,
            resize_target=model_config.resize_target
        )


def process_irradiance_data(
    user_data: pd.DataFrame,
    solar_coords: Tuple[np.ndarray, np.ndarray],
    time_array: pd.DatetimeIndex,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    data_source: DataSourceInfo,
    stage: ProcessingStage
) -> IrradianceData:
    """
    Process irradiance data either from user-provided measurements or NASA POWER data.
    Prompts the user for data source choice when needed and updates status file with the selection.

    Args:
        user_data: DataFrame containing system specifications
        solar_coords: Tuple of solar azimuth and zenith angles
        time_array: Array of timestamps for calculations
        start_dt: Start datetime
        end_dt: End datetime
        data_source: Information about data source preferences, including last file paths
        stage: Current processing stage to determine if recalculation is needed

    Returns:
        IrradianceData object containing processed irradiance data
    """
    try:
        # Only prompt if we need to recalculate or no previous choice exists
        if stage <= ProcessingStage.RAW_IRRADIANCE:
            print(f"{Fore.CYAN}What irradiance data would you like to use?")
            print(f"Options:\n"
                  "1 - Include your own data\n"
                  f"2 - Use NASA POWER data{Style.RESET_ALL}")
            use_own_data = input().strip()
            while use_own_data not in ['1', '2']:
                use_own_data = input(
                    f"{Fore.RED}Invalid choice. Please enter 1 or 2.\n{Style.RESET_ALL}")
            data_source.using_nasa = use_own_data == '2'

            # Update status.yml with the new choice
            try:
                os.makedirs('./DebugData', exist_ok=True)
                with open('status.yml', 'r') as f:
                    status_data = yaml.safe_load(f) or {}
                status_data['data_source'] = {
                    'using_nasa': data_source.using_nasa,
                    'using_day_night': data_source.using_day_night,
                    'last_irradiance_file': data_source.last_irradiance_file
                }
                with open('status.yml', 'w') as f:
                    yaml.dump(status_data, f)
                logger.info(f"Updated status.yml with data source choice: {'NASA' if data_source.using_nasa else 'CSV'}")
            except Exception as e:
                logger.warning(f"Could not update status file: {e}")
                print(f"{Fore.RED}Warning: Could not save preferences to status file{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Using {'NASA POWER' if data_source.using_nasa else 'user-provided'} irradiance data{Style.RESET_ALL}")

        if not data_source.using_nasa:
            irradiance = load_user_irradiance_data(start_dt, end_dt, data_source, solar_coords,
                                                   lat=float(user_data['Lattitude (°)'][0]),
                                                   lon=float(user_data['Longitude (°)'][0]))
        else:
            irradiance = fetch_nasa_power_data(user_data, time_array, solar_coords)

        # Validate the returned data
        if len(irradiance.direct) == 0 or len(irradiance.diffuse) == 0:
            raise ValueError(f"{Fore.RED}Irradiance data is empty{Style.RESET_ALL}")

        logger.info(f"Successfully processed irradiance data: {len(irradiance.direct)} data points")
        return irradiance

    except Exception as e:
        raise


def load_user_irradiance_data(start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                              data_source: DataSourceInfo,
                              solar_coords: Tuple[np.ndarray, np.ndarray],
                              lat: float, lon: float) -> IrradianceData:
    """
    Load and validate user-provided irradiance measurements from CSV file.
    Updates the status file with the path to the selected irradiance file.

    Args:
        start_dt: Start datetime to filter data
        end_dt: End datetime to filter data
        data_source: Information about data source preferences to update
        solar_coords: Solar position coordinates (azimuth and zenith angles)
        lat: Latitude for solar calculations
        lon: Longitude for solar calculations

    Returns:
        IrradianceData object containing processed irradiance data
    """
    while True:
        csv_files = [f for f in os.listdir("./SystemData") if f.lower().endswith('.csv')]
        if not csv_files:
            print(f"{Fore.RED}No CSV files found in './SystemData'")
            input(f"Add your irradiance data file and press Enter...{Style.RESET_ALL}")
            continue

        print(f"{Fore.CYAN}Available CSV files:{Style.RESET_ALL}")
        for idx, file in enumerate(csv_files, 1):
            print(f"  {idx}. {file}")

        try:
            choice = int(input(f"{Fore.CYAN}Select file number: {Style.RESET_ALL}")) - 1
            if not 1 <= choice <= len(csv_files):
                raise ValueError()

            try:
                with open('status.yml', 'r') as f:
                    status_data = yaml.safe_load(f) or {}
                status_data['data_source'] = {
                    'using_nasa': data_source.using_nasa,
                    'using_day_night': data_source.using_day_night,
                    'last_irradiance_file': os.path.join("./SystemData", csv_files[choice])
                }
                with open('status.yml', 'w') as f:
                    yaml.dump(status_data, f)
            except Exception as e:
                print(f"{Fore.RED}Warning: Could not update status file: {e}{Style.RESET_ALL}")

            data = process_irradiance_csv(
                filepath=os.path.join("./SystemData", csv_files[choice]),
                start_dt=start_dt,
                end_dt=end_dt,
                solar_coords=solar_coords,
                lat=lat,
                lon=lon
            )
            return data

        except (ValueError, IndexError):
            print(f"{Fore.RED}Invalid selection. Try again.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error processing file: {e}{Style.RESET_ALL}")


def process_irradiance_csv(
    filepath: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    solar_coords: Tuple[np.ndarray, np.ndarray],
    lat: float,
    lon: float
) -> IrradianceData:
    """
    Process user-provided CSV file containing irradiance measurements.

    This function reads and parses a user-provided CSV file with irradiance data,
    automatically detecting the datetime column and direct/diffuse irradiance columns.
    If columns cannot be automatically identified, it prompts the user to select them.
    It then resamples the data to match the required time range and aligns with solar coordinates.

    Args:
        filepath: Path to CSV file containing irradiance data
        start_dt: Start datetime for the analysis period
        end_dt: End datetime for the analysis period
        solar_coords: Tuple of (solar_azimuth, solar_zenith) arrays
        lat: Latitude for solar calculations and data validation
        lon: Longitude for solar calculations and data validation

    Returns:
        IrradianceData: Object containing processed direct and diffuse irradiance data,
                        solar position information, and time index

    Raises:
        ValueError: If data format is invalid or missing required columns
        RuntimeError: If user cancels column selection
    """
    # Load CSV with flexible separator detection
    df = pd.read_csv(filepath, sep=None, engine='python')

    # Check if CSV is completely empty
    if df.empty or len(df.columns) == 0:
        raise ValueError(f"{Fore.RED}CSV file is empty or has no columns: {filepath}{Style.RESET_ALL}")

    # Find datetime column
    date_col = next(
        (col for col in df.columns
         if pd.api.types.is_datetime64_any_dtype(df[col])
         or 'date' in col.lower()
         or 'time' in col.lower()),
        None
    )

    if date_col is None:
        print(f"{Fore.YELLOW}Select datetime column:{Style.RESET_ALL}")
        for idx, col in enumerate(df.columns, 1):
            print(f"  {idx}. {col}")
        while True:
            try:
                date_choice = int(
                    input(f"{Fore.CYAN}Select datetime column number: {Style.RESET_ALL}").strip())
                if not (1 <= date_choice <= len(df.columns)):
                    raise ValueError
                date_col = df.columns[date_choice - 1]
                break
            except ValueError:
                print(
                    f"{Fore.RED}Invalid choice. Please enter a number between 1 and {len(df.columns)}.{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}Using '{date_col}' as datetime column.{Style.RESET_ALL}")

    # Convert dates and set as index
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    if df.empty:
        raise ValueError(
            f"{Fore.RED}No data found between {start_dt} and {end_dt}{Style.RESET_ALL}")

    # Filter to requested date range
    mask = (df.index >= start_dt) & (df.index <= end_dt + pd.Timedelta(hours=23))
    df = df[mask]

    # Resample to hourly data if needed
    if df.index.freq != 'h':
        df = df.resample('h').mean()

    # Decide how to handle irradiance data based on available columns
    remaining_cols = [col for col in df.columns if col != date_col]

    if len(remaining_cols) == 1:
        # Single column case: Assume GHI and use Erbs decomposition
        ghi_col = remaining_cols[0]
        print(f"{Fore.YELLOW}Single data column detected: Using '{ghi_col}' as Global Horizontal Irradiance (GHI)")
        print(
            f"Will apply Erbs decomposition model to estimate direct and diffuse components.{Style.RESET_ALL}")

        ghi = df[ghi_col].values

        # Calculate solar position for decomposition
        solar_pos = get_solarposition(df.index, lat, lon)

        # Apply Erbs decomposition model
        erbs_results = erbs(ghi=ghi, zenith=solar_pos['zenith'].values, datetime_or_doy=df.index)
        direct = erbs_results['dni'].values  # Normal direct irradiance
        diffuse = erbs_results['dhi'].values  # Horizontal diffuse irradiance

    else:
        # Multiple columns: Let user choose between GHI or Direct+Diffuse
        print(f"\n{Fore.YELLOW}Available columns:{Style.RESET_ALL}")
        for idx, col in enumerate(remaining_cols, 1):
            print(f"  {idx}. {col}")
        print(f"{Fore.CYAN}\nDo you want to:")
        print("1. Use one column as GHI (will apply Erbs decomposition)")
        print(f"2. Select separate Direct and Diffuse irradiance columns{Style.RESET_ALL}")

        while True:
            choice = input(f"{Fore.CYAN}Enter choice (1/2): {Style.RESET_ALL}").strip()
            if choice in ['1', '2']:
                break
            print(f"{Fore.RED}Invalid choice. Please enter 1 or 2.{Style.RESET_ALL}")

        if choice == '1':
            # GHI case
            while True:
                try:
                    ghi_idx = int(
                        input(f"{Fore.CYAN}Select GHI column number: {Style.RESET_ALL}")) - 1
                    if 0 <= ghi_idx < len(remaining_cols):
                        ghi = df[remaining_cols[ghi_idx]].values
                        solar_pos = get_solarposition(df.index, lat, lon)
                        erbs_results = erbs(
                            ghi=ghi,
                            zenith=solar_pos['zenith'].values,
                            datetime_or_doy=df.index)
                        direct = erbs_results['dni'].values
                        diffuse = erbs_results['dhi'].values
                        break
                    print(f"{Fore.RED}Invalid column number.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        else:
            # Direct + Diffuse case
            while True:
                try:
                    direct_idx = int(
                        input(f"{Fore.CYAN}Select Direct Irradiance column number: {Style.RESET_ALL}")) - 1
                    diffuse_idx = int(
                        input(f"{Fore.CYAN}Select Diffuse Irradiance column number: {Style.RESET_ALL}")) - 1
                    if (0 <= direct_idx < len(remaining_cols) and
                        0 <= diffuse_idx < len(remaining_cols) and
                            direct_idx != diffuse_idx):
                        direct = df[remaining_cols[direct_idx]].values
                        diffuse = df[remaining_cols[diffuse_idx]].values
                        break
                    print(f"{Fore.RED}Invalid column numbers or same column selected twice.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter valid numbers.{Style.RESET_ALL}")

    return IrradianceData(
        direct=direct,
        diffuse=diffuse,
        solar_azimuth=solar_coords[0],
        solar_zenith=solar_coords[1],
        time_index=df.index
    )


def fetch_nasa_power_data(user_data: pd.DataFrame, time_array: pd.DatetimeIndex,
                          solar_coords: Tuple[np.ndarray, np.ndarray]) -> IrradianceData:
    """
    Fetch and process irradiance data from NASA POWER database.

    Args:
        user_data: DataFrame containing system specifications with latitude, longitude and time range
        time_array: Array of timestamps for aligning the data
        solar_coords: Tuple of (solar_azimuth, solar_zenith) arrays

    Returns:
        IrradianceData object containing NASA POWER direct normal and horizontal diffuse irradiance
    """
    normal_direct, hor_diffuse, _ = retrieve_NASA_POWER_irradiance(
        lat=float(user_data['Lattitude (°)'][0]),
        long=float(user_data['Longitude (°)'][0]),
        start_time=int(user_data['Start year'][0]),
        end_time=int(user_data['End year'][0])
    )

    return IrradianceData(
        direct=normal_direct,
        diffuse=hor_diffuse,
        solar_azimuth=solar_coords[0],
        solar_zenith=solar_coords[1],
        time_index=time_array
    )


def parse_time_range(
        user_data: pd.DataFrame) -> Tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    """
    Parse start and end dates from user data and create hourly time array.

    Extracts start and end years from the user data, converts them to proper datetime
    objects, and creates an hourly time range covering the entire period from
    start date at 00:00 to end date at 23:00.

    Args:
        user_data: System specifications DataFrame including 'Start year' and 'End year'
                  columns in YYYYMMDD format

    Returns:
        Tuple containing:
            - time_array: Full range of hourly timestamps for the entire period
            - start_dt: Start datetime (beginning of the start day)
            - end_dt: End datetime (beginning of the end day, without hours)

    Note:
        The time_array includes the full end date (with 23 hours added) to ensure
        complete day coverage
    """
    start_str = str(user_data['Start year'][0])
    end_str = str(user_data['End year'][0])
    start_dt = pd.to_datetime(start_str, format='%Y%m%d')
    end_dt = pd.to_datetime(end_str, format='%Y%m%d')
    time_array = pd.date_range(start=start_dt, end=end_dt + pd.Timedelta(hours=23), freq='1h')
    return time_array, start_dt, end_dt


def calculate_solar_position(user_data: pd.DataFrame,
                             time_array: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate solar position (azimuth and zenith angles) for given location and time range.

    Uses the astropy library through the sunpath_from_astropy function to calculate
    precise solar positions for each timestamp in the provided time array,
    taking into account the exact location coordinates and elevation.

    Args:
        user_data: System specifications DataFrame containing 'Longitude (°)',
                  'Lattitude (°)', and 'Elevation (m)' columns
        time_array: Array of timestamps for which to calculate solar positions

    Returns:
        Tuple containing:
            - solar_azimuth: Array of solar azimuth angles in degrees for each timestamp
            - solar_zenith: Array of solar zenith angles in degrees for each timestamp

    Note:
        The azimuth angle is measured clockwise from North (0° = North, 90° = East)
        The zenith angle is measured from the vertical (0° = directly overhead)
    """
    return sunpath_from_astropy(
        longitude=float(user_data['Longitude (°)'][0]),
        latitude=float(user_data['Lattitude (°)'][0]),
        ground_level=user_data['Elevation (m)'][0],
        time_array=time_array
    )


def handle_user_reset(control_stage: ProcessingStage) -> ProcessingStage:
    """
    Handle user request to potentially reset and redo calculations from a chosen stage.

    Args:
        control_stage (ProcessingStage): Current processing stage

    Returns:
        ProcessingStage: Updated processing stage based on user choice

    Note:
        When no changes are detected, presents a menu allowing the user to:
        1. Recalculate from Camera Calibration
        2. Recalculate from Solar Position & Irradiance
        3. Recalculate from Shading Factors
        4. Recalculate from State of Charge
        5. Continue without recalculation
    """
    if control_stage == ProcessingStage.NO_CHANGES:
        print(f"\n{Fore.CYAN}No changes detected. Would you like to:{Style.RESET_ALL}")
        print("1. Recalculate from Camera Calibration")
        print("2. Recalculate from Solar Position & Irradiance")
        print("3. Recalculate from Shading Factors")
        print("4. Recalculate from State of Charge")
        print("5. Continue without recalculation")

        while True:
            choice = input(f"{Fore.CYAN}Enter choice (1-5): {Style.RESET_ALL}").strip()

            if choice == '5':
                return ProcessingStage.NO_CHANGES
            elif choice in {'1', '2', '3', '4'}:
                stage_map = {
                    '1': ProcessingStage.CALIBRATION,
                    '2': ProcessingStage.RAW_IRRADIANCE,
                    '3': ProcessingStage.SHADING,
                    '4': ProcessingStage.STATE_OF_CHARGE
                }
                return stage_map[choice]
            else:
                print(f"{Fore.RED}Invalid choice. Please enter a number between 1 and 5.{Style.RESET_ALL}")

    return control_stage


def process_calibration(user_data: pd.DataFrame, calib_files: List[str]) -> None:
    """
    Perform camera calibration using checkerboard images.

    This function calibrates the camera using the provided checkerboard calibration images
    and saves the calibration parameters to calibration.yml file.
    
    Note: Validation of calibration parameters and images is handled by validate_calibration_images
    in read_user_data.py before this function is called.

    Args:
        user_data: DataFrame containing system specifications including
                   'Calib vertex short', 'Calib vertex long', 'Calib square size (mm)'
        calib_files: List of paths to validated calibration images

    Returns:
        None: The calibration results are saved to a file
    """
    try:
        # Extract calibration parameters (already validated in validate_calibration_images)
        pattern_cols = int(user_data['Calib vertex short'][0])
        pattern_rows = int(user_data['Calib vertex long'][0])
        square_size = float(user_data['Calib square size (mm)'][0])

        logger.info(f"Starting calibration with {len(calib_files)} images, "
                   f"pattern: {pattern_cols}x{pattern_rows}, square size: {square_size}mm")
        
        calibrate_camera(
            pattern_cols=pattern_cols,
            pattern_rows=pattern_rows,
            square_size=square_size,
            images=calib_files
        )
        
        # Verify calibration file was created
        if not os.path.exists('./calibration.yml'):
            raise RuntimeError(f"Calibration completed but calibration.yml file was not created")
        
        print(f"{Fore.GREEN}Camera calibration completed successfully! Calibration parameters saved to calibration.yml{Style.RESET_ALL}")
        logger.info("Camera calibration completed successfully")

    except Exception:
        raise


def compute_compensated_irradiance(
    irradiance: IrradianceData,
    shading: Tuple[np.ndarray, np.ndarray],
    save_path: str = './DebugData/irradiance.csv'
) -> np.ndarray:
    """
    Calculate shading-compensated irradiance and save results to CSV.
    Applies shading factors to direct and diffuse irradiance components and
    saves both raw and compensated values to a CSV file.

    Args:
        irradiance: Original irradiance data containing direct and diffuse components
        shading: Tuple of (direct_shading, diffuse_shading) factors from 0-1
        save_path: Path to save the CSV output (default: './DebugData/irradiance.csv')

    Returns:
        np.ndarray: Final combined irradiance values (compensated direct + compensated diffuse)
    """
    # Calculate total irradiance with shading factors applied
    direct_shading, diffuse_shading = shading
    compensated_direct = np.multiply(irradiance.direct, 1 - direct_shading)
    compensated_diffuse = np.multiply(irradiance.diffuse, 1 - diffuse_shading)
    final_irradiance = compensated_direct + compensated_diffuse

    # Save shading-processed irradiance data
    shaded_data = {
        'Direct_irradiance': irradiance.direct,
        'Diffuse_irradiance': irradiance.diffuse,
        'Compensated_direct': compensated_direct,
        'Compensated_diffuse': compensated_diffuse,
    }

    shaded_df = pd.DataFrame(shaded_data, index=irradiance.time_index)
    shaded_df.index.name = 'Timeseries'
    shaded_df.fillna(0, inplace=True)

    # Save to CSV for debugging/reference
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shaded_df.to_csv(save_path)
    print(f"{Fore.GREEN}Shading-processed irradiance data saved to: {save_path}{Style.RESET_ALL}")

    return final_irradiance


def load_compensated_irradiance(
        file_path: str = './DebugData/irradiance.csv') -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load previously calculated shading-compensated irradiance from CSV.

    Args:
        file_path: Path to the CSV file containing shading-processed irradiance data

    Returns:
        Tuple containing:
            - np.ndarray: Final combined irradiance values (compensated_direct + compensated_diffuse)
            - pd.DatetimeIndex: Time index from the loaded data

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file doesn't contain the expected data columns
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{Fore.RED}Shaded irradiance file not found: {file_path}{Style.RESET_ALL}")

        # Load the CSV file
        df = pd.read_csv(file_path)

        # Check if the file has a datetime index
        if 'Timeseries' in df.columns:
            df['Timeseries'] = pd.to_datetime(df['Timeseries'])
            df = df.set_index('Timeseries')
        else:
            raise ValueError(f"{Fore.RED}Missing Timeseries column in irradiance data{Style.RESET_ALL}")

        # Validate required columns exist
        required_columns = ['Compensated_direct', 'Compensated_diffuse']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{Fore.RED}Missing required columns in shaded irradiance data: {missing_columns}{Style.RESET_ALL}")

        compensated_direct = df['Compensated_direct'].values
        compensated_diffuse = df['Compensated_diffuse'].values

        # Check for NaN values
        if np.isnan(compensated_direct).any() or np.isnan(compensated_diffuse).any():
            logger.warning("NaN values found in compensated irradiance data, filling with zeros")
            compensated_direct = np.nan_to_num(compensated_direct, 0)
            compensated_diffuse = np.nan_to_num(compensated_diffuse, 0)

        # Combine direct and diffuse components
        final_irradiance = compensated_direct + compensated_diffuse

        print(f"{Fore.GREEN}Successfully loaded shading-compensated irradiance data from: {file_path}{Style.RESET_ALL}")
        logger.info(f"Loaded {len(final_irradiance)} irradiance data points from {file_path}")
        
        return final_irradiance

    except Exception:
        raise


def process_shading(
    user_data: pd.DataFrame,
    irradiance: IrradianceData,
    im_height: int,
    im_width: int,
    flag_combination: bool,
    img: np.ndarray,
    data_source: DataSourceInfo
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate shading factors from sky images.
    Processes sky image(s) to create a binary mask and calculates both direct and diffuse shading
    factors using camera calibration parameters.

    Args:
        user_data: System specifications with orientation and inclination data
        irradiance: Solar irradiance data with azimuth, zenith and timestamps
        im_height: Image height in pixels
        im_width: Image width in pixels
        flag_combination: Whether to use multiple images (batch processing)
        img: The sky image as numpy array (if not using multiple images)
        data_source: Data source information to determine irradiance type (NASA vs user)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Direct and diffuse shading factors as arrays from 0-1
        where 1 means fully shaded (100% loss)
    """
    # Get camera parameters
    poly_incident_angle_to_radius, principal_point, estimated_fov = import_camera_intrinsic_function()
    # Get model configuration from user
    model_config = select_model_configuration()
    # Process sky image
    try:
        skyimage_bw_mask = process_sky_image(
            model_config=model_config,
            flag_combination=flag_combination,
            img=img
        )
        
        # Check if sky image processing failed
        if skyimage_bw_mask is None:
            raise RuntimeError(f"Sky image processing returned None - check model files and image processing steps")
    except Exception as e:
        # Re-raise with more context about what was being attempted
        raise RuntimeError(f"Sky image processing failed during model inference: {str(e)}") from e

    # Calculate shading factors
    diffuse_shading = compute_diffuse_shading_factor(
        image=skyimage_bw_mask * 255,
        poly_incident_angle_to_radius=poly_incident_angle_to_radius,
        principal_point=principal_point,
        estimated_fov=estimated_fov,
        im_height=im_height,
        im_width=im_width,
        image_orientation=float(user_data['Image orientation (°)'][0]),
        image_inclination=float(user_data['Image inclination (°)'][0]),
        inclined_surface_orientation=float(user_data['Plane orientation (°)'][0]),
        inclined_surface_inclination=float(user_data['Plane inclination (°)'][0])
    )

    # Choose irradiance type based on data source
    # NASA POWER provides Direct Normal Irradiance (DNI)
    # User data provides Direct Horizontal Irradiance (DHI) or is converted to it using Erbs
    irradiance_type = 'normal' if data_source.using_nasa else 'horizontal'

    direct_shading = compute_direct_shading_factor_generic(
        image=skyimage_bw_mask * 255,
        im_height=im_height,
        im_width=im_width,
        poly_incident_angle_to_radius=poly_incident_angle_to_radius,
        principal_point=principal_point,
        image_orientation=float(user_data['Image orientation (°)'][0]),
        image_inclination=float(user_data['Image inclination (°)'][0]),
        estimated_fov=estimated_fov,
        az_zen_array=[irradiance.solar_azimuth, irradiance.solar_zenith],
        original_time_array=irradiance.time_index,
        inclined_surface_orientation=float(user_data['Plane orientation (°)'][0]),
        inclined_surface_inclination=float(user_data['Plane inclination (°)'][0]),
        irradiance_type=irradiance_type
    )

    return direct_shading, diffuse_shading


def process_state_of_charge(
    user_data: pd.DataFrame,
    final_irradiance: np.ndarray,
    time_index: pd.DatetimeIndex,
    data_source: DataSourceInfo
) -> None:
    """
    Calculate battery state of charge over time based on irradiance and system specs.
    Uses either hourly or day/night consumption profiles based on data_source preferences.

    Args:
        user_data: System specifications with battery and panel parameters
        final_irradiance: Combined compensated irradiance values (W/m²)
        time_index: DatetimeIndex corresponding to the irradiance values
        data_source: Information about data source preferences, including profile type

    Returns:
        None: Results are saved to files by the called functions
    """

    # Print which profile type we're using
    print(f"{Fore.GREEN}Using {'day/night' if data_source.using_day_night else 'hourly'} consumption profile{Style.RESET_ALL}")

    # Extract system parameters
    params = {
        'panel_power': float(user_data['Solar panel peak wattage (W)'][0]),
        'converter_eff': float(user_data['Converter efficiency (%)'][0]),
        'converter_power': float(user_data['Converter max power (W)'][0]),
        'charge_eff': float(user_data['Charge efficiency (%)'][0]),
        'discharge_eff': float(user_data['Discharge efficiency (%)'][0]),
        'max_soc': float(user_data['Max SOC (%)'][0]),
        'min_soc': float(user_data['Min SOC (%)'][0]),
        'batt_capacity': float(user_data['Batt nominal capacity (Ah)'][0]),
        'batt_voltage': float(user_data['Batt nominal voltage (V)'][0])
    }

    # Calculate state of charge based on profile type
    if not data_source.using_day_night:
        state_of_charge_estimation(
            final_irradiance,
            time_index,
            params['panel_power'],
            params['converter_eff'],
            params['converter_power'],
            params['charge_eff'],
            params['discharge_eff'],
            params['max_soc'],
            params['min_soc'],
            params['batt_capacity'],
            params['batt_voltage']
        )
    else:
        state_of_charge_estimation_day_night(
            final_irradiance,
            time_index,
            params['panel_power'],
            params['converter_eff'],
            params['converter_power'],
            params['charge_eff'],
            params['discharge_eff'],
            params['max_soc'],
            params['min_soc'],
            params['batt_capacity'],
            params['batt_voltage']
        )


def initialize_error_logging() -> None:
    """
    Initialize error logging system and create necessary directories.
    
    Sets up file-based logging to capture detailed error information during
    program execution. Creates the DebugData directory if it doesn't exist,
    tests write permissions, and configures logging handlers. Falls back to
    console-only logging if file operations fail.
    
    Note:
        The global logger variable is updated after configuration
    """
    global logger
    
    try:
        # Create DebugData directory if it doesn't exist
        os.makedirs('./DebugData', exist_ok=True)
        
        # Test write permissions
        test_file = './DebugData/.test_write'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        # Configure logging now that directory exists
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./DebugData/solar_estimation_errors.log', mode='w')  # Overwrite log file each run
            ],
            force=True  # Override any existing configuration
        )
        
        # Reinitialize logger after configuration
        logger = logging.getLogger(__name__)
        
        # Log startup information
        logger.info("="*60)
        logger.info("Solar Estimation System Started")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info("="*60)
        
    except Exception as e:
        print(f"{Fore.RED}Warning: Could not initialize error logging: {e}")
        print(f"Continuing without file logging...{Style.RESET_ALL}")
        
        # Fallback to console-only logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[],
            force=True
        )
        logger = logging.getLogger(__name__)


def get_user_continue_choice() -> bool:
    """
    Get user choice to continue or exit after completion.
    
    Prompts the user to either exit the program or run another estimation
    cycle, providing a clean way to handle program termination or restart.
    
    Returns:
        bool: True to continue with another estimation, False to exit
    """
    while True:
        choice = input(f"\n{Fore.CYAN}Options:\n{Style.RESET_ALL}"
                       "1 - Exit program\n"
                       "2 - Run another estimation\n"
                       f"{Fore.CYAN}Choice: {Style.RESET_ALL}").strip().lower()

        if choice == '1':
            print(f"\n{Fore.BLUE}Thank you for using the solar estimation system!{Style.RESET_ALL}")
            return False
        elif choice == '2':
            print(f"\n{Fore.CYAN}Starting new estimation...{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")


def get_user_recovery_choice() -> bool:
    """
    Get user choice for error recovery after an exception occurs.
    
    Provides options for handling errors: either restart the program to
    try again or quit completely. Also directs users to the log file
    for detailed error information.
    
    Returns:
        bool: True to restart the program, False to quit
    """
    print(f"\n{Fore.YELLOW}For detailed error information, check the log file at: './DebugData/solar_estimation_errors.log'{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}RECOVERY OPTIONS:{Style.RESET_ALL}")
    print("1. Restart the program")
    print("2. Quit the program")
    
    while True:
        user_choice = input(f"{Fore.CYAN}Enter choice (1-2): {Style.RESET_ALL}").strip()
        if user_choice == '1':
            return True  # Restart the program
        elif user_choice == '2':
            return False  # Quit the program
        else:
            print(f"{Fore.RED}Invalid choice. Please enter 1 or 2.{Style.RESET_ALL}")


def execute_calibration_step(stage: ProcessingStage, user_data: pd.DataFrame, calib_files: List[str]) -> None:
    """
    Execute camera calibration step if needed.
    
    Conditionally performs camera calibration based on the current processing stage.
    Only runs calibration if the stage indicates it's needed, providing efficient
    pipeline execution by skipping unnecessary steps.
    
    Args:
        stage: Current processing stage to determine if calibration is needed
        user_data: System specifications containing calibration parameters
        calib_files: List of validated calibration image file paths
        
    Raises:
        Exception: Re-raises any calibration errors after handling
    """
    if stage <= ProcessingStage.CALIBRATION:
        print(f"\n{Fore.CYAN}=== Step 1: Camera Calibration ==={Style.RESET_ALL}")
        try:
            process_calibration(user_data, calib_files)
        except Exception as e:
            handle_calibration_error(e)
            raise


def get_time_and_solar_data(stage: ProcessingStage, user_data: pd.DataFrame) -> Tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate time array and solar coordinates if needed.
    
    Conditionally computes time range and solar position data based on the
    current processing stage. Skips calculation if no changes detected.
    
    Args:
        stage: Current processing stage to determine if calculation is needed
        user_data: System specifications containing time range and location data
        
    Returns:
        Tuple containing time array, start/end timestamps, and solar coordinates,
        or (None, None, None, None) if calculation is skipped
        
    Raises:
        Exception: Re-raises any time/solar calculation errors after handling
    """
    if stage < ProcessingStage.NO_CHANGES:
        try:
            time_array, start_dt, end_dt = parse_time_range(user_data)
            solar_coords = calculate_solar_position(user_data, time_array)
            return time_array, start_dt, end_dt, solar_coords
        except Exception as e:
            handle_time_solar_error(e)
            raise
    return None, None, None, None


def execute_irradiance_step(stage: ProcessingStage, user_data: pd.DataFrame, solar_coords: Tuple[np.ndarray, np.ndarray], 
                           time_array: pd.DatetimeIndex, start_dt: pd.Timestamp, end_dt: pd.Timestamp, 
                           data_source: DataSourceInfo) -> Tuple[IrradianceData, ProcessingStage]:
    """
    Execute irradiance calculation step.
    
    Processes solar irradiance data either by calculating from NASA POWER API
    or loading from user-provided CSV files. Handles both fresh calculation
    and loading of previously processed data based on the processing stage.
    
    Args:
        stage: Current processing stage to determine action needed
        user_data: System specifications
        solar_coords: Solar azimuth and zenith angle arrays
        time_array: Array of timestamps for the analysis period
        start_dt: Start datetime for the analysis
        end_dt: End datetime for the analysis
        data_source: Information about data source preferences
        
    Returns:
        Tuple containing processed irradiance data and updated processing stage,
        or (None, updated_stage) if restart is needed
        
    Raises:
        Exception: Re-raises any irradiance processing errors after handling
    """
    if stage <= ProcessingStage.RAW_IRRADIANCE:
        print(f"\n{Fore.CYAN}=== Step 2: Solar Position and Irradiance ==={Style.RESET_ALL}")
        try:
            irradiance_data = process_irradiance_data(
                user_data=user_data,
                solar_coords=solar_coords,
                time_array=time_array,
                start_dt=start_dt,
                end_dt=end_dt,
                data_source=data_source,
                stage=stage
            )
            return irradiance_data, stage
        except Exception as e:
            handle_irradiance_error(e, data_source)
            raise
    elif stage < ProcessingStage.NO_CHANGES:
        # Try to load saved irradiance data
        try:
            if data_source.using_nasa:
                print(f"{Fore.YELLOW}Using NASA POWER data...{Style.RESET_ALL}")
                irradiance_data = fetch_nasa_power_data(user_data, time_array, solar_coords)
            else:
                print(f"{Fore.YELLOW}Loading saved irradiance data...{Style.RESET_ALL}")
                irradiance_data = process_irradiance_csv(
                    data_source.last_irradiance_file,
                    start_dt, end_dt, solar_coords,
                    float(user_data['Lattitude (°)'][0]),
                    float(user_data['Longitude (°)'][0])
                )
            return irradiance_data, stage
        except Exception as e:
            new_stage = handle_saved_irradiance_error(e)
            return None, new_stage  # Signal to restart
    
    return None, stage


def execute_shading_step(stage: ProcessingStage, user_data: pd.DataFrame, irradiance_data: IrradianceData,
                        im_height: int, im_width: int, flag_combination: bool, 
                        img: np.ndarray, data_source: DataSourceInfo) -> Tuple[np.ndarray, ProcessingStage]:
    """
    Execute shading calculation step.
    
    Processes sky images to calculate shading factors and apply them to irradiance data.
    Uses deep learning models for sky segmentation and geometric calculations for
    shading factor computation. Handles both fresh calculation and loading of
    previously processed results.
    
    Args:
        stage: Current processing stage to determine action needed
        user_data: System specifications with orientation parameters
        irradiance_data: Solar irradiance and position data
        im_height: Sky image height in pixels
        im_width: Sky image width in pixels
        flag_combination: Whether to combine multiple sky images
        img: Sky image array (if using single image)
        data_source: Information about data source and irradiance type
        
    Returns:
        Tuple containing final compensated irradiance array and updated processing stage,
        or (None, updated_stage) if restart is needed
        
    Raises:
        Exception: Re-raises any shading calculation errors after handling
    """
    if stage <= ProcessingStage.SHADING:
        print(f"\n{Fore.CYAN}=== Step 3: Calculate Shading ==={Style.RESET_ALL}")
        try:
            direct_shading, diffuse_shading = process_shading(
                user_data=user_data,
                irradiance=irradiance_data,
                im_height=im_height,
                im_width=im_width,
                flag_combination=flag_combination,
                img=img,
                data_source=data_source
            )

            # Calculate and save shading-compensated irradiance immediately
            print(f"{Fore.YELLOW}Calculating shading-compensated irradiance...{Style.RESET_ALL}")
            final_irradiance = compute_compensated_irradiance(
                irradiance=irradiance_data,
                shading=(direct_shading, diffuse_shading)
            )
            return final_irradiance, stage
        except Exception as e:
            handle_shading_error(e)
            raise
    elif stage == ProcessingStage.STATE_OF_CHARGE:
        # If skipping directly to state of charge, try to load previously calculated irradiance
        print(f"\n{Fore.CYAN}=== Skipping Step 3: Attempting to load previous shading results ==={Style.RESET_ALL}")
        try:
            final_irradiance = load_compensated_irradiance()
            return final_irradiance, stage
        except (FileNotFoundError, ValueError) as e:
            new_stage = handle_shading_results_error(e)
            return None, new_stage  # Signal to restart

    return None, stage


def execute_soc_step(stage: ProcessingStage, user_data: pd.DataFrame, final_irradiance: np.ndarray,
                    time_index: pd.DatetimeIndex, data_source: DataSourceInfo) -> None:
    """
    Execute state of charge calculation step.
    
    Performs battery state of charge simulation using the shading-compensated
    irradiance data and system parameters. Chooses between hourly and day/night
    consumption profiles based on user preferences.
    
    Args:
        stage: Current processing stage to determine if calculation is needed
        user_data: System specifications with battery and solar panel parameters
        final_irradiance: Final compensated solar irradiance values (W/m²)
        time_index: DatetimeIndex corresponding to the irradiance values
        data_source: Information about consumption profile type
        
    Raises:
        Exception: Re-raises any battery modeling errors after handling
    """
    if stage <= ProcessingStage.STATE_OF_CHARGE:
        print(f"\n{Fore.CYAN}=== Step 4: Calculate State of Charge ==={Style.RESET_ALL}")
        try:
            process_state_of_charge(
                user_data=user_data,
                final_irradiance=final_irradiance,
                time_index=time_index,
                data_source=data_source
            )
        except Exception as e:
            handle_battery_error(e)
            raise