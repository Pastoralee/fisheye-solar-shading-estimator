# Fisheye Solar Shading Estimator

A comprehensive solar estimation system that uses fisheye camera images to calculate shading factors and estimate solar irradiance for photovoltaic systems.

## About This Work

This project was developed as part of academic research detailed in the thesis:

**"[Photovoltaic applications in demanding situations : estimation and optimisation of solar ressources for autonomous power supplies]"** - Available at: https://theses.hal.science/tel-04725060v1

The sky segmentation methodology used in this work is the subject of an upcoming research paper that will be published soon.

For detailed instructions on how to run the program, including guidance on taking sky images, camera calibration procedures, and system setup, please refer to the `README.docx` file included in this repository.

## Features

- **Camera Calibration**: Automatic calibration of fisheye cameras using the omnicalib library
- **Solar Position Calculation**: Uses Astropy for accurate solar position calculations
- **Sky Segmentation**: Advanced machine learning-based sky detection
- **Shading Analysis**: Computes direct and diffuse shading factors from sky images
- **Irradiance Estimation**: Retrieves NASA POWER irradiance data and applies shading corrections
- **Battery State of Charge**: Estimates battery performance with day/night consumption profiles

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

1. Clone this repository:
```bash
git clone https://github.com/Pastoralee/fisheye-solar-shading-estimator.git
cd fisheye-solar-shading-estimator
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Download ML Model Weights

**Required**: Download the pre-trained machine learning models for sky segmentation:

1. Access the Google Drive folder: https://drive.google.com/drive/folders/1PnKakX55PCW72MTsl-TXBb6TM5EOUejA
2. Download all files from the `Models` folder
3. Place the downloaded model files in the `SystemData/` directory of this project

### Additional Setup

- **Calibration images**: Place calibration images in the `CalibrationImages/` directory
- **Sky images**: Place site sky images for analysis in the `SkyImageOfSite/` directory
- **System data**: Configure system specifications in the `SystemData/` directory using the provided Excel templates

## Usage

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

## Model Files

The sky segmentation functionality requires pre-trained machine learning models that must be downloaded separately from Google Drive and placed in the `SystemData/` directory.

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
├── README.docx               # Detailed usage instructions
└── *.py                       # Main program files
```

## Citation

If you use this work in your research, please cite the associated thesis:

```bibtex
@phdthesis{Cao2023,
  title = {{Photovoltaic applications in demanding situations : estimation and optimisation of solar ressources for autonomous power supplies}},
  author = {Cao, Kha Bao Khanh},
  url = {https://theses.hal.science/tel-04725060},
  school = {{INSA de Toulouse}},
  year = {2023}
}
```

*Note: A research paper on the sky segmentation methodology is forthcoming and will be available for citation.*

## Contributing

Please feel free to submit issues and pull requests to improve this project.
