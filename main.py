"""
Solar Estimation System Runner
==============================

This module contains the main execution function and workflow orchestration
for the solar estimation system. It imports all necessary functions from
the main module and provides a clean entry point.

Usage:
    python main.py
"""

from colorama import Fore, Style
from core import (
    ProcessingStage,
    initialize_error_logging,
    show_welcome_message,
    handle_user_reset,
    execute_calibration_step,
    get_time_and_solar_data,
    execute_irradiance_step,
    execute_shading_step,
    execute_soc_step,
    get_user_continue_choice,
    get_user_recovery_choice,
    handle_data_loading_error,
    read_user_data
)


def main():
    """
    Main execution loop for solar estimation system.

    Orchestrates the entire solar estimation pipeline:
    1. Loads and validates user data, including system specifications and images
    2. Performs camera calibration if needed
    3. Calculates solar position and irradiance data
    4. Computes shading factors from sky images
    5. Applies shading to irradiance data
    6. Calculates battery state of charge over time

    Handles different restart scenarios based on detected changes or user choices,
    with options to skip steps when appropriate.
    """
    initialize_error_logging()
    show_welcome_message()
    forced_stage = None  # Track if we're forcing a specific stage

    while True:
        try:
            # Load and validate all user data (only if not forcing a stage)
            if forced_stage is None:
                try:
                    (user_data, calib_files, stage, im_height, im_width,
                     flag_combination, img, data_source) = read_user_data()
                except Exception as err:
                    handle_data_loading_error(err)
                    raise
            else:
                # Use the forced stage and keep existing user data
                stage = forced_stage
                forced_stage = None  # Reset after using it

            input(f"{Fore.CYAN}Press Enter to begin estimation...{Style.RESET_ALL}")

            # Check if user wants to redo calculations (only if not using forced stage)
            if forced_stage is None:
                stage = handle_user_reset(stage)
            
            # Step 1: Camera Calibration
            execute_calibration_step(stage, user_data, calib_files)

            if stage < ProcessingStage.NO_CHANGES:
                # Calculate time_array and solar_coords as they're needed for irradiance
                time_array, start_dt, end_dt, solar_coords = get_time_and_solar_data(stage, user_data)

                # Step 2: Calculate Solar Position and Irradiance
                irradiance_data, stage = execute_irradiance_step(
                    stage, user_data, solar_coords, time_array, start_dt, end_dt, data_source
                )

                # Check if we need to restart due to irradiance loading failure
                if irradiance_data is None:
                    forced_stage = ProcessingStage.RAW_IRRADIANCE
                    continue  # Restart the main loop with forced stage

                # Step 3: Calculate Shading Factors
                final_irradiance, stage = execute_shading_step(
                    stage, user_data, irradiance_data, im_height, im_width, flag_combination, img, data_source
                )

                # Check if we need to restart due to shading loading failure
                if final_irradiance is None:
                    forced_stage = ProcessingStage.SHADING
                    continue  # Restart the main loop with forced stage

                # Step 4: Calculate State of Charge
                execute_soc_step(
                    stage, user_data, final_irradiance,
                    irradiance_data.time_index, data_source
                )

            print(f"\n{Fore.GREEN}All calculations completed successfully!{Style.RESET_ALL}")

            # Ask user if they want to continue
            if not get_user_continue_choice():
                return

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Program interrupted by user.{Style.RESET_ALL}")
            return
        except Exception:
            # Log the error if needed, but do not use unused variable
            if not get_user_recovery_choice():
                return
            continue


if __name__ == '__main__':
    main()
