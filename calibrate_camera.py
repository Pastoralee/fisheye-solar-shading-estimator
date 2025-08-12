import omnicalib as omni
from omnicalib.chessboard import get_points
import cv2
import numpy as np
from typing import List, Dict
import torch
from colorama import Fore, Style
from import_camera_intrinsic_function import import_camera_intrinsic_function


def calibrate_camera(
        pattern_cols: int,
        pattern_rows: int,
        square_size: float,
        images: List[str]) -> None:
    """
    Calibrate a camera using checkerboard patterns.

    This function performs camera calibration using multiple images of a checkerboard pattern.
    It combines OpenCV's corner detection with OmniCalib's fisheye calibration algorithm.

    Args:
        pattern_cols: Number of inner corners along the checkerboard columns
        pattern_rows: Number of inner corners along the checkerboard rows
        square_size: Physical size of each checkerboard square in millimeters
        images: List of paths to calibration images

    Raises:
        AssertionError: If fewer than 8 usable calibration images are found
    """
    # Define checkerboard parameters
    CHECKERBOARD = (pattern_cols, pattern_rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 2)

    # Initialize point storage
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Create object points template
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Prepare OmniCalib data structures
    object_points = get_points(pattern_rows, pattern_cols, float(square_size)).view(-1, 3)
    detections: Dict[str, Dict] = {}
    nb_images_used_for_calib = 0

    print(f"{Fore.YELLOW}Running through your calibration image set and detecting corners...{Style.RESET_ALL}")

    def sharpen_image(img: np.ndarray) -> np.ndarray:
        """Apply sharpening filter to enhance corner detection."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    for fname in images:
        # Load and preprocess image
        img_bw = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        img_bw = sharpen_image(img_bw)  # Initial sharpening

        # Downsize image for faster processing
        downsized_img_bw = cv2.resize(
            img_bw,
            (int(img_bw.shape[1] / 8), int(img_bw.shape[0] / 8)),
            interpolation=cv2.INTER_AREA
        )
        downsized_img_bw = sharpen_image(downsized_img_bw)  # Additional sharpening

        # Attempt to find checkerboard corners in downsized image
        ret, temp_corners = cv2.findChessboardCorners(
            downsized_img_bw,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # Try alternative method if first attempt fails
        if not ret:
            ret, temp_corners = cv2.findChessboardCornersSB(
                downsized_img_bw,
                CHECKERBOARD,
                cv2.CALIB_CB_EXHAUSTIVE
            )

        if ret:
            objpoints.append(objp)

            # Scale corners back to original image size
            corners = temp_corners * 8  # Multiply by 8 since image was downsized by 8

            # Refine corner positions
            corners2 = cv2.cornerSubPix(
                img_bw,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            # Store calibration points
            imgpoints.append(corners2)

            # Store data for OmniCalib
            detections[fname] = {
                'image_points': torch.from_numpy(corners2).to(torch.float64).squeeze(1),
                'object_points': object_points
            }

            nb_images_used_for_calib += 1
            print(f"{Fore.MAGENTA}Detected corners in {fname}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Unable to detect corners in {fname}{Style.RESET_ALL}")

    # Print calibration summary
    print(f"{Fore.LIGHTMAGENTA_EX}Total number of images: {len(images)}")
    print(f"Total images with detected checkerboards: {nb_images_used_for_calib}{Style.RESET_ALL}")

    # Validate sufficient calibration images
    if nb_images_used_for_calib < 8:
        raise AssertionError(
            f"{Fore.RED}Below 8 images used for calibration, the result would not be precise{Style.RESET_ALL}")

    print(f"{Fore.GREEN}Corner detection complete!{Style.RESET_ALL}")

    # Run OmniCalib calibration
    print(f"{Fore.LIGHTYELLOW_EX}Starting calibration using OmniCalib")
    print(
        f"Module courtesy of Thomas Pönitz, Github: https://github.com/tasptz/py-omnicalib{Style.RESET_ALL}")

    omni.main(
        detections=detections,
        degree=4,
        threshold=100,
        count=round(nb_images_used_for_calib / 4)
    )
    # Update calibration file with FOV information
    import_camera_intrinsic_function()
