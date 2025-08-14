from typing import Tuple, Dict
import os
import pandas as pd
import numpy as np
from colorama import Fore, Style
import matplotlib.pyplot as plt
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
from config import PATHS


def save_debug_data(debug_data: pd.DataFrame, filename: str) -> None:
    """
    Save debug data to Excel file with retry on file lock conflicts.
    
    Args:
        debug_data: DataFrame containing debug data to save
        filename: Name of the Excel file to save (will be saved in debug_data folder)
        
    Note:
        Prompts user to close file if it's locked and retries automatically
    """
    while True:
        try:
            import os
            os.makedirs(PATHS["debug_data"], exist_ok=True)
            debug_data.to_excel(os.path.join(PATHS["debug_data"], filename))
            break
        except PermissionError:
            input(
                f"{Fore.RED}It seems {filename} is opened, close it and press ENTER to retry...{Style.RESET_ALL}")


def plot_soc_evolution(
    time_array: pd.DatetimeIndex,
    final_irradiance: np.ndarray,
    soc_evolution: np.ndarray,
    output_path: str
) -> None:
    """
    Create and save a plot showing irradiance and state of charge evolution over time.
    
    Args:
        time_array: Time index for the data
        final_irradiance: Solar irradiance values (Wh/m²)
        soc_evolution: State of charge percentages over time
        output_path: File path to save the plot
        
    Note:
        Creates a dual-axis plot with irradiance on left axis and SOC on right axis
    """
    plt.rcParams.update({'font.size': 8})
    _, ax = plt.subplots()

    # Plot irradiance
    ax.step(time_array, final_irradiance, color='orange', linewidth=2)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_xlim([time_array[0], time_array[-1]])
    ax.set_ylabel('Hourly solar irradiation (Whm-2)', color='orange', fontsize=16)
    ax.set_ylim([0, 1000])

    # Plot SOC on secondary axis
    ax2 = ax.twinx()
    ax2.step(time_array, soc_evolution, color='blue', linewidth=2)
    ax2.set_ylim([0, 100])
    ax2.set_ylabel('Hourly state of charge (%)', color='blue', fontsize=16)

    plt.grid()
    plt.savefig(output_path)
    print(f"{Fore.GREEN}Plot saved to {output_path}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTCYAN_EX}Close figure to continue...{Style.RESET_ALL}")
    plt.show()


def save_verdict(minimum_soc: float, min_soc: float, suffix: str = "") -> None:
    """
    Save system operation verdict to a timestamped text file.
    
    Args:
        minimum_soc: The minimum SOC reached during the simulation period
        min_soc: The minimum allowable SOC threshold for the system
        suffix: Optional suffix to append to the filename
        
    Note:
        Creates a verdict about whether the system can operate continuously
        based on SOC analysis
    """
    verdict = (
        f"Min SOC throughout observation period is {round(minimum_soc, 2)}% "
        f"which is {'higher than' if minimum_soc > min_soc else 'equal to'} "
        f"the minimum SOC of {min_soc}%\n"
        f"So the system {'COULD' if minimum_soc > min_soc else 'MAY NOT'} OPERATE CONTINUOUSLY!"
    )

    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    filename = f"{timestamp}-verdict{suffix}.txt"

    with open(filename, "w") as f:
        f.write(verdict)

    color = Fore.GREEN if minimum_soc > min_soc else Fore.RED
    print(f"{color}{verdict}{Style.RESET_ALL}")


def calculate_energy_flows(
    final_irradiance: np.ndarray,
    consumption_over_time: np.ndarray,
    solar_peak: float,
    conv_eff: float,
    conv_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate energy flows in the solar power system.
    
    Args:
        final_irradiance: Solar irradiance values (W/m²)
        consumption_over_time: Power consumption values (W)
        solar_peak: Solar panel peak power rating (W)
        conv_eff: Converter efficiency (%)
        conv_max: Maximum converter power output (W)
        
    Returns:
        Tuple containing:
        - Energy output from converter (W)
        - Net energy flow to/from battery (W, positive=charging, negative=discharging)
    """
    # Convert irradiance to converter output
    energy_out_of_converter = final_irradiance * solar_peak * conv_eff / (1000 * 100)
    energy_out_of_converter = np.minimum(energy_out_of_converter, conv_max)

    # Calculate net energy flow to/from battery
    energy_flow_to_from_battery = energy_out_of_converter - consumption_over_time

    return energy_out_of_converter, energy_flow_to_from_battery


def calculate_soc_evolution(
    energy_flow: np.ndarray,
    initial_soc: float,
    max_soc: float,
    min_soc: float,
    batt_nom_cap: float,
    batt_nom_volt: float,
    charge_eff: float,
    discharge_eff: float
) -> np.ndarray:
    """
    Calculate battery state of charge evolution over time.
    
    Args:
        energy_flow: Net energy flow to/from battery (W, positive=charging)
        initial_soc: Initial state of charge (%)
        max_soc: Maximum allowable state of charge (%)
        min_soc: Minimum allowable state of charge (%)
        batt_nom_cap: Battery nominal capacity (Ah)
        batt_nom_volt: Battery nominal voltage (V)
        charge_eff: Charging efficiency (%)
        discharge_eff: Discharging efficiency (%)
        
    Returns:
        np.ndarray: State of charge values over time (%)
        
    Note:
        SOC is clamped to the min/max limits and accounts for charge/discharge efficiencies
    """
    # Convert to numpy array if it's a pandas Series to avoid FutureWarning
    energy_flow = np.asarray(energy_flow)
    
    soc_evolution = np.zeros(len(energy_flow))
    soc_evolution[0] = initial_soc

    for i in range(1, len(energy_flow)):
        # Calculate new SOC based on energy flow direction
        if energy_flow[i] >= 0:
            # Charging
            soc_evolution[i] = (
                soc_evolution[i - 1] +
                energy_flow[i] * (charge_eff / 100) * 100 / (batt_nom_cap * batt_nom_volt)
            )
        else:
            # Discharging
            soc_evolution[i] = (
                soc_evolution[i - 1] +
                energy_flow[i] * 100 / (batt_nom_cap * batt_nom_volt * (discharge_eff / 100))
            )

        # Clamp SOC to limits
        soc_evolution[i] = np.clip(soc_evolution[i], min_soc, max_soc)

    return soc_evolution


def state_of_charge_estimation(
    final_irradiance: np.ndarray,
    time_array: pd.DatetimeIndex,
    solar_peak: float,
    conv_eff: float,
    conv_max: float,
    charge_eff: float,
    discharge_eff: float,
    max_soc: float,
    min_soc: float,
    batt_nom_cap: float,
    batt_nom_volt: float
) -> np.ndarray:
    """
    Calculate state of charge evolution based on fixed hourly consumption profile.

    Args:
        final_irradiance: Solar irradiance values
        time_array: Array of timestamps
        solar_peak: Peak solar power output
        conv_eff: Converter efficiency (%)
        conv_max: Maximum converter output
        charge_eff: Battery charging efficiency (%)
        discharge_eff: Battery discharging efficiency (%)
        max_soc: Maximum state of charge (%)
        min_soc: Minimum state of charge (%)
        batt_nom_cap: Battery nominal capacity
        batt_nom_volt: Battery nominal voltage

    Returns:
        np.ndarray: Array of state of charge values over time
    """
    print(f'{Fore.YELLOW}Computing the estimated evolution of the system state of charge...{Style.RESET_ALL}')

    # Load and process consumption profile
    raw_consumption = pd.read_excel(PATHS["consumption_profile"], index_col=None)
    consumption_over_time = np.zeros(len(time_array))

    for _, row in raw_consumption.iterrows():
        mask = time_array.hour == row['Hour of day']
        consumption_over_time[mask] = row['Consumption (Wh)']

    # Calculate energy flows
    _, energy_flow_to_from_battery = calculate_energy_flows(
        final_irradiance, consumption_over_time, solar_peak, conv_eff, conv_max
    )

    # Calculate SOC evolution
    soc_evolution = calculate_soc_evolution(
        energy_flow_to_from_battery, max_soc, max_soc, min_soc,
        batt_nom_cap, batt_nom_volt, charge_eff, discharge_eff
    )

    print(f'{Fore.GREEN}Done!{Style.RESET_ALL}')

    # Save results to Excel
    print(f'{Fore.YELLOW}Saving debug data to Excel and generating visualization...{Style.RESET_ALL}')
    debug_data = pd.DataFrame({
        'Time': time_array.tz_localize(None),
        'SOC': soc_evolution,
        'Energy flow': energy_flow_to_from_battery,
        'Consumption': consumption_over_time,
        'Irradiation': final_irradiance
    }).set_index('Time')

    save_debug_data(debug_data, 'soc_ev.xlsx')

    # Generate and save visualization
    plot_soc_evolution(
        time_array,
        final_irradiance,
        soc_evolution,
        os.path.join(PATHS["debug_data"], 'hourly_soc_estimation_from_user_visual.png')
    )

    # Save system operation verdict
    minimum_soc_throughout = np.min(soc_evolution)
    save_verdict(minimum_soc_throughout, min_soc)


def calculate_sun_times(
    dates: pd.DatetimeIndex,
    lat: float,
    lon: float,
    timezone: str = 'UTC'
) -> Tuple[Dict, Dict]:
    """Calculate sunrise and sunset times for given dates and location.

    Args:
        dates: Array of dates to calculate sun times for
        lat: Latitude in degrees
        lon: Longitude in degrees
        timezone: Timezone name (default: 'UTC')

    Returns:
        Tuple[Dict, Dict]: Dictionaries mapping dates to sunrise and sunset hours
    """
    sunrise_dict = {}
    sunset_dict = {}
    location = EarthLocation(lat=lat, lon=lon)

    for date in dates:
        # Create hourly times for the day (0 to 23 hours)
        times = pd.date_range(
            start=pd.Timestamp(date),
            end=pd.Timestamp(date) + pd.Timedelta(hours=23),
            freq='1h',
            tz=timezone
        )

        # Calculate sun positions
        astropy_times = Time(times.to_pydatetime())
        frame = AltAz(obstime=astropy_times, location=location)
        sun_altaz = get_sun(astropy_times).transform_to(frame)
        altitudes = sun_altaz.alt.degree

        # Find sunrise and sunset
        above_horizon = np.where(altitudes > 0)[0]
        if len(above_horizon) == 0:
            sunrise_hour, sunset_hour = 6, 18  # fallback values
        else:
            sunrise_hour = times[above_horizon[0]].hour
            sunset_hour = times[above_horizon[-1]].hour

        sunrise_dict[pd.Timestamp(date).date()] = sunrise_hour
        sunset_dict[pd.Timestamp(date).date()] = sunset_hour

    return sunrise_dict, sunset_dict


def get_consumption_profile(sunrise_dict: Dict, sunset_dict: Dict, time_array: pd.DatetimeIndex,
                            day_consumption: float, night_consumption: float) -> np.ndarray:
    """Generate consumption profile based on day/night periods.

    Args:
        sunrise_dict: Dictionary mapping dates to sunrise hours
        sunset_dict: Dictionary mapping dates to sunset hours
        time_array: Array of timestamps
        day_consumption: Consumption during day hours (Wh)
        night_consumption: Consumption during night hours (Wh)

    Returns:
        np.ndarray: Array of consumption values for each timestamp
    """
    consumption_over_time = np.zeros(len(time_array))

    for i, t in enumerate(time_array):
        date = t.date()
        hour = t.hour
        sunrise = sunrise_dict[date]
        sunset = sunset_dict[date]

        # Day consumption is from sunrise to 1 hour after sunset
        if sunrise <= hour <= sunset + 1:
            consumption_over_time[i] = day_consumption
        else:
            consumption_over_time[i] = night_consumption

    return consumption_over_time


def state_of_charge_estimation_day_night(
    final_irradiance: np.ndarray,
    time_array: pd.DatetimeIndex,
    solar_peak: float,
    conv_eff: float,
    conv_max: float,
    charge_eff: float,
    discharge_eff: float,
    max_soc: float,
    min_soc: float,
    batt_nom_cap: float,
    batt_nom_volt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate state of charge evolution using day/night consumption profile.

    This function uses astronomical calculations to determine sunrise and sunset
    times for each day, and applies different consumption profiles for day and
    night periods.

    Args:
        final_irradiance: Solar irradiance values
        time_array: Array of timestamps
        solar_peak: Peak solar power output
        conv_eff: Converter efficiency (%)
        conv_max: Maximum converter output
        charge_eff: Battery charging efficiency (%)
        discharge_eff: Battery discharging efficiency (%)
        max_soc: Maximum state of charge (%)
        min_soc: Minimum state of charge (%)
        batt_nom_cap: Battery nominal capacity
        batt_nom_volt: Battery nominal voltage

    Returns:
        Tuple[np.ndarray, np.ndarray]: SOC evolution and consumption over time
    """
    print(f'{Fore.YELLOW}Computing the estimated evolution of the system state of charge '
          f'(day/night profile, astropy sunrise/sunset)...{Style.RESET_ALL}')
    # Load location data
    user_data = pd.read_excel(PATHS["system_specs"], index_col=None)
    lat = float(user_data['Lattitude (°)'][0])
    lon = float(user_data['Longitude (°)'][0])

    # Load consumption profiles
    profile = pd.read_excel(PATHS["day_night_profile"], index_col=None)
    day_consumption = profile['Day Consumption (Wh)'][0]
    night_consumption = profile['Night Consumption (Wh)'][0]

    # Calculate sunrise/sunset times
    dates = pd.to_datetime(time_array.date).unique()
    sunrise_dict, sunset_dict = calculate_sun_times(dates, lat, lon)

    # Generate consumption profile
    consumption_over_time = get_consumption_profile(
        sunrise_dict, sunset_dict, time_array,
        day_consumption, night_consumption
    )

    # Calculate energy flows
    energy_out_of_converter, energy_flow_to_from_battery = calculate_energy_flows(
        final_irradiance, consumption_over_time, solar_peak, conv_eff, conv_max
    )

    # Calculate SOC evolution
    soc_evolution = calculate_soc_evolution(
        energy_flow_to_from_battery, max_soc, max_soc, min_soc,
        batt_nom_cap, batt_nom_volt, charge_eff, discharge_eff
    )

    # Check for NaN values in energy calculations
    nan_mask = np.isnan(energy_out_of_converter) | np.isnan(consumption_over_time)
    if np.any(nan_mask):
        print(f"{Fore.RED}Warning: NaN values detected in calculations at times: "
              f"{time_array[nan_mask]}{Style.RESET_ALL}")

    print(f'{Fore.GREEN}Done!{Style.RESET_ALL}')

    # Save debug data
    print(f'{Fore.YELLOW}Saving debug data and generating visualization...{Style.RESET_ALL}')
    debug_data = pd.DataFrame({
        'Time': time_array.tz_localize(None),
        'SOC': soc_evolution,
        'Energy flow': energy_flow_to_from_battery,
        'Consumption': consumption_over_time,
        'Irradiation': final_irradiance
    }).set_index('Time')

    save_debug_data(debug_data, 'soc_ev_day_night.xlsx')

    # Generate and save visualization
    plot_soc_evolution(
        time_array,
        final_irradiance,
        soc_evolution,
        os.path.join(PATHS["debug_data"], 'hourly_soc_estimation_day_night.png')
    )

    # Save system operation verdict
    minimum_soc_throughout = np.min(soc_evolution)
    save_verdict(minimum_soc_throughout, min_soc, "_day_night")

    return soc_evolution, consumption_over_time
