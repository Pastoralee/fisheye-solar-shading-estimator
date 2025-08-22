import os
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from colorama import Fore, Style
from calibrate_camera import calibrate_camera
from compute_diffuse_shading_factor import compute_diffuse_shading_factor
from compute_direct_shading_factor import compute_direct_shading_factor_generic
from config import MODEL_CONFIGS, PATHS, DataSourceInfo, ModelConfig, ProcessingStage
from data_loader import (
    load_calibration_images,
    load_consumption_profile,
    load_custom_irradiance_data,
    load_sky_images,
    load_system_specs,
)
from import_camera_intrinsic_function import import_camera_intrinsic_function
from inference import batch_disk_mask_inference, inference
from retrieve_NASA_POWER_irradiance import retrieve_NASA_POWER_irradiance
from stage_manager import StageManager
from state_of_charge_estimation import (
    state_of_charge_estimation,
    state_of_charge_estimation_day_night,
)
from sunpath_from_astropy import sunpath_from_astropy
from validation import get_user_choice


class SolarEstimationPipeline:
    """Main processing pipeline for solar estimation.
    
    This class orchestrates the complete solar shading estimation workflow,
    including camera calibration, solar position calculation, sky image processing,
    shading factor computation, and battery state of charge modeling.
    
    Attributes:
        user_data: System specifications loaded from Excel file
        data_source: Information about data sources used (NASA/custom files)
        sky_image: Loaded sky image for processing
        calib_files: List of calibration image file paths
        stage: Current processing stage
        stage_manager: Manager for intelligent stage processing
        all_images: List of all loaded image file paths
        flag_combination: Whether to combine multiple images
        direct_irradiance_type: Type of direct irradiance calculation
    """

    def __init__(self) -> None:
        """Initialize the solar estimation pipeline with default values.
        
        Sets up all pipeline components including data containers, stage management,
        and processing flags. All data attributes start as None or empty and are
        populated during pipeline execution.
        """
        self.user_data: Optional[pd.DataFrame] = None
        self.data_source: DataSourceInfo = DataSourceInfo()
        self.sky_image: Optional[np.ndarray] = None
        self.calib_files: List[str] = []
        self.stage: ProcessingStage = ProcessingStage.NO_CHANGES
        self.stage_manager: StageManager = StageManager()
        self.all_images: List[str] = []
        self.flag_combination: bool = False
        self.direct_irradiance_type: str = "normal"  # Default for NASA data

    def load_all_data(self) -> None:
        """Load and validate all user data.
        
        Loads system specifications, consumption profiles, sky images, and 
        calibration images. Validates that all images have consistent dimensions.
        
        Raises:
            ValueError: If image dimensions don't match between sky and calibration images
        """
        print(f"{Fore.YELLOW}Loading system data...{Style.RESET_ALL}")

        # Load system specifications
        self.user_data = load_system_specs()

        # Load consumption profile
        _, self.data_source = load_consumption_profile()

        # Load sky images
        self.sky_image, self.flag_combination, self.all_images = load_sky_images()

        # Load calibration images (with dimension validation against sky images)
        self.calib_files = load_calibration_images(self.all_images)

        print(f"{Fore.GREEN}All data loaded successfully{Style.RESET_ALL}")

    def get_model_config(self) -> ModelConfig:
        """Get model configuration from user input.
        
        Presents available model options to the user and returns the selected
        configuration for sky image processing.
        
        Returns:
            ModelConfig: Selected model configuration containing model name,
                        LGBM usage flag, and resize target dimensions
        """
        print(f"{Fore.CYAN}Select model configuration:{Style.RESET_ALL}")
        print("1. 512x512 EfficientNet-b5 (fastest)")
        print("2. 512x512 EfficientNet-b7")
        print("3. Base EfficientNet-b5")
        print("4. Base EfficientNet-b7")
        print("5. Base EfficientNet-b5 + LGBM")
        print("6. Base EfficientNet-b7 + LGBM (best quality)")

        choice = get_user_choice("Enter choice (1-6): ", ["1", "2", "3", "4", "5", "6"])
        return MODEL_CONFIGS[choice]

    def run_calibration(self) -> None:
        """Run camera calibration using chessboard images.
        
        Performs camera calibration using the provided calibration images
        and saves the camera parameters to a YAML file.
        
        Raises:
            Exception: If calibration fails due to insufficient or invalid images
        """
        print(f"{Fore.YELLOW}Running camera calibration...{Style.RESET_ALL}")
        
        # Ensure DebugData directory exists for calibration outputs
        os.makedirs(PATHS["debug_data"], exist_ok=True)
        
        try:
            calibrate_camera(
                pattern_cols=int(self.user_data["Calib vertex short"][0]),
                pattern_rows=int(self.user_data["Calib vertex long"][0]),
                square_size=float(self.user_data["Calib square size (mm)"][0]),
                images=self.calib_files,
            )
            print(f"{Fore.GREEN}Camera calibration completed{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Calibration failed: {e}{Style.RESET_ALL}")
            raise

    def get_solar_data(self) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
        """Calculate time range and solar positions.
        
        Generates a datetime array based on the specified date range and calculates
        solar azimuth and zenith angles for each timestamp using astronomical calculations.
        
        Returns:
            Tuple containing:
                - pd.DatetimeIndex: Hourly timestamp array for the date range
                - np.ndarray: Solar azimuth angles in degrees
                - np.ndarray: Solar zenith angles in degrees
        """
        # Parse time range
        start_str = str(self.user_data["Start year"][0])
        end_str = str(self.user_data["End year"][0])
        start_dt = pd.to_datetime(start_str, format="%Y%m%d")
        end_dt = pd.to_datetime(end_str, format="%Y%m%d")
        time_array = pd.date_range(start=start_dt, end=end_dt + pd.Timedelta(hours=23), freq="1h")

        # Calculate solar positions
        solar_azimuth, solar_zenith = sunpath_from_astropy(
            longitude=float(self.user_data["Longitude (°)"][0]),
            latitude=float(self.user_data["Lattitude (°)"][0]),
            ground_level=self.user_data["Elevation (m)"][0],
            time_array=time_array,
        )

        return time_array, solar_azimuth, solar_zenith

    def get_irradiance_data(
        self, time_array: pd.DatetimeIndex, solar_azimuth: np.ndarray, solar_zenith: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Get irradiance data from NASA or user file.
        
        Allows user to choose between NASA POWER database or custom CSV/Excel files
        for irradiance data. Handles data loading, validation, and alignment with
        the time array.
        
        Args:
            time_array: Array of timestamps for solar calculations
            solar_azimuth: Solar azimuth angles in degrees
            solar_zenith: Solar zenith angles in degrees
            
        Returns:
            Tuple containing:
                - np.ndarray: Direct irradiance values
                - np.ndarray: Diffuse irradiance values  
                - np.ndarray: Updated solar azimuth angles
                - np.ndarray: Updated solar zenith angles
                - pd.DatetimeIndex: Updated time array
        """
        print(f"{Fore.CYAN}Choose irradiance data source:{Style.RESET_ALL}")
        print("1. NASA POWER database (recommended)")
        print("2. User-provided .csv/.xls/.xlsx file")

        choice = get_user_choice("Enter choice (1/2): ", ["1", "2"])

        if choice == "1":
            self.data_source.using_nasa = True
            try:
                normal_direct, hor_diffuse, _ = retrieve_NASA_POWER_irradiance(
                    lat=float(self.user_data["Lattitude (°)"][0]),
                    long=float(self.user_data["Longitude (°)"][0]),
                    start_time=int(self.user_data["Start year"][0]),
                    end_time=int(self.user_data["End year"][0]),
                )
                return normal_direct, hor_diffuse, solar_azimuth, solar_zenith, time_array
            except Exception as e:
                print(f"{Fore.RED}Error fetching NASA data: {e}{Style.RESET_ALL}")
                raise
        else:
            # Load custom irradiance file from SystemData folder
            self.data_source.using_nasa = False
            print(f"{Fore.YELLOW}Loading custom irradiance file...{Style.RESET_ALL}")

            datetime_col, direct_irr, diffuse_irr, irr_type = load_custom_irradiance_data(
                time_array,
                lat=float(self.user_data["Lattitude (°)"][0]),
                lon=float(self.user_data["Longitude (°)"][0]),
            )

            if datetime_col is None:
                print(f"{Fore.RED}Failed to load irradiance data{Style.RESET_ALL}")
                raise ValueError("Failed to load custom irradiance data")

            # Store the irradiance type for shading calculations
            self.direct_irradiance_type = irr_type  # 'normal' or 'horizontal'

            print(
                f"{Fore.GREEN}Custom irradiance data loaded successfully ({irr_type} type){Style.RESET_ALL}"
            )
            return direct_irr, diffuse_irr, solar_azimuth, solar_zenith, time_array

    def calculate_shading(
        self, solar_azimuth: np.ndarray, solar_zenith: np.ndarray, time_array: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate direct and diffuse shading factors.
        
        Uses machine learning models to process sky images and calculate
        shading factors for direct and diffuse solar radiation.
        
        Args:
            solar_azimuth: Array of solar azimuth angles in degrees
            solar_zenith: Array of solar zenith angles in degrees
            time_array: DatetimeIndex array corresponding to the solar data
            
        Returns:
            Tuple containing:
                - np.ndarray: Direct shading factors (0-1, where 0 = fully shaded)
                - np.ndarray: Diffuse shading factors (0-1, where 0 = fully shaded)
        """
        # Get model configuration
        model_config = self.get_model_config()

        # Process sky image
        if self.flag_combination:
            skyimage_bw_mask = batch_disk_mask_inference(
                self.all_images,
                model_config.model_name,
                model_config.use_lgbm,
                model_config.resize_target,
            )
        else:
            skyimage_bw_mask = inference(
                self.sky_image,
                model_config.model_name,
                model_config.use_lgbm,
                model_config.resize_target,
            )

        # Get camera parameters
        poly_incident_angle_to_radius, principal_point, estimated_fov = (
            import_camera_intrinsic_function()
        )

        # Calculate shading factors
        im_height, im_width = skyimage_bw_mask.shape

        diffuse_shading = compute_diffuse_shading_factor(
            image=skyimage_bw_mask * 255,
            poly_incident_angle_to_radius=poly_incident_angle_to_radius,
            principal_point=principal_point,
            estimated_fov=estimated_fov,
            im_height=im_height,
            im_width=im_width,
            image_orientation=float(self.user_data["Image orientation (°)"][0]),
            image_inclination=float(self.user_data["Image inclination (°)"][0]),
            inclined_surface_orientation=float(self.user_data["Plane orientation (°)"][0]),
            inclined_surface_inclination=float(self.user_data["Plane inclination (°)"][0]),
        )

        direct_shading = compute_direct_shading_factor_generic(
            image=skyimage_bw_mask * 255,
            im_height=im_height,
            im_width=im_width,
            poly_incident_angle_to_radius=poly_incident_angle_to_radius,
            principal_point=principal_point,
            image_orientation=float(self.user_data['Image orientation (°)'][0]),
            image_inclination=float(self.user_data['Image inclination (°)'][0]),
            estimated_fov=estimated_fov,
            az_zen_array=[solar_azimuth, solar_zenith],
            original_time_array=time_array,
            inclined_surface_orientation=float(self.user_data['Plane orientation (°)'][0]),
            inclined_surface_inclination=float(self.user_data['Plane inclination (°)'][0]),
            irradiance_type=self.direct_irradiance_type
        )

        return direct_shading, diffuse_shading

    def calculate_final_irradiance(
        self, 
        direct_irr: np.ndarray, 
        diffuse_irr: np.ndarray, 
        direct_shading: np.ndarray, 
        diffuse_shading: np.ndarray, 
        time_array: pd.DatetimeIndex
    ) -> np.ndarray:
        """Apply shading factors to irradiance and save results.
        
        Applies calculated shading factors to direct and diffuse irradiance
        values and saves the results to a CSV file for later use.
        
        Args:
            direct_irr: Direct irradiance values
            diffuse_irr: Diffuse irradiance values
            direct_shading: Direct shading factors (0-1)
            diffuse_shading: Diffuse shading factors (0-1)
            time_array: Corresponding timestamps
            
        Returns:
            np.ndarray: Final compensated irradiance values (direct + diffuse)
        """
        compensated_direct = np.multiply(direct_irr, 1 - direct_shading)
        compensated_diffuse = np.multiply(diffuse_irr, 1 - diffuse_shading)
        final_irradiance = compensated_direct + compensated_diffuse

        # Save results
        os.makedirs(PATHS["debug_data"], exist_ok=True)
        results_df = pd.DataFrame(
            {
                "Direct_irradiance": direct_irr,
                "Diffuse_irradiance": diffuse_irr,
                "Compensated_direct": compensated_direct,
                "Compensated_diffuse": compensated_diffuse,
            },
            index=time_array,
        )
        results_df.to_csv(PATHS["irradiance_csv"])

        print(f"{Fore.GREEN}Final irradiance calculated and saved{Style.RESET_ALL}")
        return final_irradiance

    def calculate_soc(self, final_irradiance: np.ndarray, time_array: pd.DatetimeIndex) -> None:
        """Calculate battery state of charge using consumption profiles.
        
        Uses either hourly consumption profiles or day/night patterns to model
        battery charging and discharging behavior based on solar irradiance.
        
        Args:
            final_irradiance: Final compensated irradiance values
            time_array: Corresponding timestamps
            
        Raises:
            Exception: If state of charge calculation fails
        """
        print(f"{Fore.YELLOW}Calculating battery state of charge...{Style.RESET_ALL}")

        try:
            if self.data_source.using_day_night:
                state_of_charge_estimation_day_night(
                    final_irradiance=final_irradiance,
                    time_array=time_array,
                    solar_peak=float(self.user_data['Solar panel peak wattage (W)'][0]),
                    conv_eff=float(self.user_data['Converter efficiency (%)'][0]),
                    conv_max=float(self.user_data['Converter max power (W)'][0]),
                    charge_eff=float(self.user_data['Charge efficiency (%)'][0]),
                    discharge_eff=float(self.user_data['Discharge efficiency (%)'][0]),
                    max_soc=float(self.user_data['Max SOC (%)'][0]),
                    min_soc=float(self.user_data['Min SOC (%)'][0]),
                    batt_nom_cap=float(self.user_data['Batt nominal capacity (Ah)'][0]),
                    batt_nom_volt=float(self.user_data['Batt nominal voltage (V)'][0])
                )
            else:
                state_of_charge_estimation(
                    final_irradiance=final_irradiance,
                    time_array=time_array,
                    solar_peak=float(self.user_data['Solar panel peak wattage (W)'][0]),
                    conv_eff=float(self.user_data['Converter efficiency (%)'][0]),
                    conv_max=float(self.user_data['Converter max power (W)'][0]),
                    charge_eff=float(self.user_data['Charge efficiency (%)'][0]),
                    discharge_eff=float(self.user_data['Discharge efficiency (%)'][0]),
                    max_soc=float(self.user_data['Max SOC (%)'][0]),
                    min_soc=float(self.user_data['Min SOC (%)'][0]),
                    batt_nom_cap=float(self.user_data['Batt nominal capacity (Ah)'][0]),
                    batt_nom_volt=float(self.user_data['Batt nominal voltage (V)'][0])
                )
        except Exception as e:
            print(f"{Fore.RED}Battery modeling failed: {e}{Style.RESET_ALL}")
            raise

    def _recalculate_shading_stage(self, required_stages: List[ProcessingStage]) -> None:
        """Helper method to recalculate shading stage when irradiance data is missing.
        
        Resets the shading stage and adds it back to the required stages list
        if not already present. Used for error recovery when irradiance files
        are missing or corrupted.
        
        Args:
            required_stages: List of stages that need to be executed, modified in-place
        """
        self.stage_manager.reset_stage(ProcessingStage.SHADING)
        # Add shading stage back to required stages if not already there
        if ProcessingStage.SHADING not in required_stages:
            shading_index = len(required_stages)
            for i, s in enumerate(required_stages):
                if s > ProcessingStage.SHADING:
                    shading_index = i
                    break
            required_stages.insert(shading_index, ProcessingStage.SHADING)

    def run_complete_pipeline(self) -> bool:
        """Run the complete estimation pipeline with intelligent stage management.
        
        Orchestrates the entire solar shading estimation workflow using intelligent
        stage management to avoid unnecessary recomputation. Handles user interaction
        for configuration choices and error recovery.
        
        Returns:
            bool: True if pipeline completed successfully, False otherwise
            
        Raises:
            ValueError: If required data is missing or invalid
            Exception: If any processing stage fails
        """
        print(f"{Fore.CYAN}=== Solar Estimation System ==={Style.RESET_ALL}")

        # Always load data first to determine what files we're working with
        self.load_all_data()

        # Define stage input files for change detection
        stage_inputs = {
            ProcessingStage.CALIBRATION: self.calib_files,
            ProcessingStage.RAW_IRRADIANCE: self.all_images,
            ProcessingStage.SHADING: [PATHS["calibration_file"]] + self.all_images,
            ProcessingStage.STATE_OF_CHARGE: [
                PATHS["consumption_profile"],
                PATHS["day_night_profile"],
            ],
        }

        # Define expected output files for each stage
        stage_outputs = {
            ProcessingStage.CALIBRATION: [PATHS["calibration_file"]],
            ProcessingStage.RAW_IRRADIANCE: [],  # No specific output file path defined
            ProcessingStage.SHADING: [PATHS["irradiance_csv"]],
            ProcessingStage.STATE_OF_CHARGE: [],  # No specific output file path defined
        }

        # Determine required stages
        required_stages = self.stage_manager.get_required_stages(stage_inputs, self.user_data)

        # If no changes detected, ask user what to do (old behavior)
        if not required_stages:
            print(f"\n{Fore.GREEN}No changes detected. All stages up to date.{Style.RESET_ALL}")
            print(self.stage_manager.get_stage_status_summary())

            print(f"\n{Fore.CYAN}Would you like to:{Style.RESET_ALL}")
            print("1. Recalculate from Camera Calibration")
            print("2. Recalculate from Solar Position & Irradiance")
            print("3. Recalculate from Shading Factors")
            print("4. Recalculate from State of Charge")
            print("5. Continue without recalculation")

            choice = get_user_choice("Enter choice (1-5): ", ["1", "2", "3", "4", "5"])

            stage_map = {
                "1": ProcessingStage.CALIBRATION,
                "2": ProcessingStage.RAW_IRRADIANCE,
                "3": ProcessingStage.SHADING,
                "4": ProcessingStage.STATE_OF_CHARGE,
                "5": ProcessingStage.NO_CHANGES,
            }

            selected_stage = stage_map[choice]
            if selected_stage != ProcessingStage.NO_CHANGES:
                self.stage_manager.reset_stage(selected_stage)
                required_stages = [
                    s for s in ProcessingStage
                    if s >= selected_stage and s != ProcessingStage.NO_CHANGES
                ]
            else:
                print(f"{Fore.GREEN}No processing needed. Exiting.{Style.RESET_ALL}")
                return True

        print(
            f"\n{Fore.YELLOW}Stages to execute: {[s.name for s in required_stages]}{Style.RESET_ALL}"
        )

        # Execute required stages
        time_array = None
        solar_azimuth = None
        solar_zenith = None
        direct_irr = None
        diffuse_irr = None
        final_irradiance = None

        for stage in required_stages:
            try:
                if stage == ProcessingStage.CALIBRATION:
                    print(f"\n{Fore.CYAN}=== Stage 1: Camera Calibration ==={Style.RESET_ALL}")
                    self.run_calibration()
                    self.stage_manager.mark_stage_completed(
                        stage, stage_inputs[stage], stage_outputs[stage]
                    )

                elif stage == ProcessingStage.RAW_IRRADIANCE:
                    print(
                        f"\n{Fore.CYAN}=== Stage 2: Solar Position & Irradiance ==={Style.RESET_ALL}"
                    )
                    time_array, solar_azimuth, solar_zenith = self.get_solar_data()
                    direct_irr, diffuse_irr, solar_azimuth, solar_zenith, time_array = (
                        self.get_irradiance_data(time_array, solar_azimuth, solar_zenith)
                    )
                    self.stage_manager.mark_stage_completed(
                        stage, stage_inputs[stage], stage_outputs[stage]
                    )

                elif stage == ProcessingStage.SHADING:
                    print(f"\n{Fore.CYAN}=== Stage 3: Shading Calculation ==={Style.RESET_ALL}")
                    # Need solar data if not already computed
                    if time_array is None:
                        time_array, solar_azimuth, solar_zenith = self.get_solar_data()
                        direct_irr, diffuse_irr, solar_azimuth, solar_zenith, time_array = (
                            self.get_irradiance_data(time_array, solar_azimuth, solar_zenith)
                        )

                    direct_shading, diffuse_shading = self.calculate_shading(
                        solar_azimuth, solar_zenith, time_array
                    )
                    final_irradiance = self.calculate_final_irradiance(
                        direct_irr, diffuse_irr, direct_shading, diffuse_shading, time_array
                    )
                    self.stage_manager.mark_stage_completed(
                        stage, stage_inputs[stage], stage_outputs[stage]
                    )

                elif stage == ProcessingStage.STATE_OF_CHARGE:
                    print(f"\n{Fore.CYAN}=== Stage 4: Battery Modeling ==={Style.RESET_ALL}")
                    # Need final irradiance if not already computed
                    if final_irradiance is None:
                        # Try to load from saved file, if missing, redo shading stage
                        if os.path.exists(PATHS["irradiance_csv"]):
                            try:
                                irr_df = pd.read_csv(
                                    PATHS["irradiance_csv"], index_col=0, parse_dates=True
                                )
                                final_irradiance = (
                                    irr_df["Compensated_direct"] + irr_df["Compensated_diffuse"]
                                )
                                time_array = irr_df.index
                            except Exception as e:
                                print(f"{Fore.YELLOW}Cannot load previous irradiance data: {e}")
                                print(f"Will recalculate shading stage...{Style.RESET_ALL}")
                                self._recalculate_shading_stage(required_stages)
                                continue
                        else:
                            print(
                                f"{Fore.YELLOW}Irradiance data not found. Will recalculate shading stage...{Style.RESET_ALL}"
                            )
                            self._recalculate_shading_stage(required_stages)
                            continue

                    self.calculate_soc(final_irradiance, time_array)
                    self.stage_manager.mark_stage_completed(
                        stage, stage_inputs[stage], stage_outputs[stage]
                    )

                print(f"{Fore.GREEN}{stage.name} completed successfully{Style.RESET_ALL}")

            except Exception:
                # Reset this stage and all subsequent stages
                self.stage_manager.reset_stage(stage)
                raise

        return True
