# Fisheye Solar Shading Estimator

A comprehensive solar estimation system that uses fisheye camera images to calculate shading factors and estimate solar irradiance for autonomous photovoltaic systems. This tool processes sky images captured with fisheye lenses to determine how obstacles (buildings, trees, etc.) affect solar panel performance throughout the year.

## About This Work

This project was developed as part of academic research detailed in the thesis:

**"Photovoltaic applications in demanding situations: estimation and optimisation of solar resources for autonomous power supplies"** by Cao, Kha Bao Khanh (2023) - Available at: https://theses.hal.science/tel-04725060v1

The sky segmentation methodology used in this work employs advanced machine learning models (EfficientNet-based architectures with optional LightGBM refinement) for accurate sky/obstacle detection in fisheye images.

## Key Features

- **Intelligent Pipeline Management**: Stage-based processing with change detection to avoid unnecessary recomputation
- **Fisheye Camera Calibration**: Automatic calibration using chessboard patterns with the integrated omnicalib library
- **Precise Solar Position Calculation**: Uses Astropy library for astronomical accuracy in solar tracking
- **Advanced Sky Segmentation**: Multiple EfficientNet-based ML models with optional LightGBM post-processing
- **Dual Shading Analysis**: Separate computation of direct and diffuse solar radiation shading factors
- **Flexible Irradiance Data Sources**: Supports both NASA POWER API and custom CSV/Excel irradiance files
- **Battery State of Charge Modeling**: Comprehensive battery performance simulation with configurable consumption profiles
- **Data Validation & Error Handling**: Robust input validation and graceful error recovery

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Compatible camera for fisheye image capture (for data collection)

### Step 1: Clone Repository

```bash
git clone https://github.com/Pastoralee/fisheye-solar-shading-estimator.git
cd fisheye-solar-shading-estimator
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Machine Learning Models

**Critical Step**: Download the pre-trained models for sky segmentation:

1. **Access Google Drive**: https://drive.google.com/drive/folders/1PnKakX55PCW72MTsl-TXBb6TM5EOUejA
2. **Download from Models folder**: Download all model files (.pt and .txt files)
3. **Place in SystemData/**: Move all downloaded model files to the `SystemData/` directory

**Available Models:**
- `efficientnet-b5.pt` / `efficientnet-b7.pt` (Main segmentation models)
- `meta_model_b5.txt` / `meta_model_b7.txt` (LightGBM refinement models)

### Step 4: Prepare Input Data

1. **Calibration Images**: Place chessboard calibration images in `CalibrationImages/`
2. **Sky Images**: Place site fisheye images for analysis in `SkyImageOfSite/`
3. **System Configuration**: Configure system parameters in `SystemData/` using provided Excel templates:
   - `System_Specifications.xlsx` (Location, camera setup, solar system specs)
   - `Consumption_Profile.xlsx` (Hourly energy consumption profile)
   - `Day_Night_Profile.xlsx` (Simplified day/night consumption patterns)

## Usage

### Quick Start

Run the main program:
```bash
python main.py
```

The program will guide you through:
1. Loading and validating user data and system specifications
2. Camera calibration (if needed)
3. Solar position and irradiance data calculation
4. Shading factor computation from sky images using ML models
5. Battery state of charge estimation

## Directory Structure

```
├── CalibrationImages/          # Camera calibration images
├── DebugData/                  # Debug output files
├── SkyImageOfSite/            # Site sky images for analysis
├── SystemData/                # System specifications, profiles, and ML models
│   ├── Consumption_Profile.xlsx
│   ├── Day_Night_Profile.xlsx
│   ├── System_Specifications.xlsx
│   └── [ML model files]       # Download from Google Drive
├── omnicalib/                 # Camera calibration library
├── README.pdf               # Detailed usage instructions
└── *.py                       # Main program files
```

## Citation

If you use this work in your research, please cite the associated thesis:

```bibtex
@phdthesis{Cao2023,
  title = {{Photovoltaic applications in demanding situations: estimation and optimisation of solar resources for autonomous power supplies}},
  author = {Cao, Kha Bao Khanh},
  url = {https://theses.hal.science/tel-04725060},
  school = {{INSA de Toulouse}},
  year = {2023}
}
```

*Note: A research paper on the sky segmentation methodology is forthcoming and will be available for citation.*

## Contributing

Please feel free to submit issues and pull requests to improve this project.
