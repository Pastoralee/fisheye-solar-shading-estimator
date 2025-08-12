from typing import Tuple, List, Dict, Optional
import os
import cv2
import numpy as np
import yaml as yml
import torch
import lightgbm as lgb
from scipy.ndimage import uniform_filter, generic_filter
import segmentation_models_pytorch as smp
from camera_coords_to_image_intrinsic import camera_coords_to_image_intrinsic
from import_camera_intrinsic_function import import_camera_intrinsic_function


def estimate_radius(pprad_path: str) -> None:
    """
    Estimate the radius of the fisheye lens and save the parameters to a YAML file.

    This function calculates the radius of the fisheye lens based on the camera's intrinsic
    parameters and field of view. The results are saved to a YAML file containing both the
    principal point and the calculated radius.

    Args:
        pprad_path (str): Path to save the YAML file containing principal point and radius.
    """
    poly_incident_angle_to_radius, principal_point, estimated_fov = import_camera_intrinsic_function()

    theta = np.deg2rad([estimated_fov])
    x_prime = np.tan(theta)
    fov_limit = camera_coords_to_image_intrinsic(np.column_stack(
        (x_prime, [0])), poly_incident_angle_to_radius, principal_point)

    distance_to_fov = fov_limit[0][0] - principal_point[0]
    radius = round(distance_to_fov) + 1
    data = {'principal_point': principal_point, 'radius': radius}
    with open(pprad_path, 'w') as f:
        yml.safe_dump(data, f)


def crop_around_disk(pprad_path: str, img: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Crop the image around the fisheye lens disk and apply a circular mask.

    This function reads camera parameters from a YAML file, crops the image around the
    fisheye lens disk, and applies a circular mask to remove non-disk areas. It ensures
    the cropping stays within image boundaries and maintains aspect ratio.

    Args:
        pprad_path (str): Path to the YAML file containing principal point and radius.
        img (np.ndarray): Input image array of shape (H, W, C).

    Returns:
        Tuple[np.ndarray, List[int]]: A tuple containing:
            - The cropped and masked image
            - List [x_min, x_max, y_min, y_max] defining the crop boundaries
    """
    with open(pprad_path, 'r') as f:
        data = yml.load(f, Loader=yml.SafeLoader)
    principal_point = data['principal_point']
    radius = data['radius']

    cx, cy = map(round, principal_point)
    img_height, img_width = img.shape[:2]
    x_min = max(cx - radius, 0)
    y_min = max(cy - radius, 0)
    x_max = min(cx + radius, img_width - 1)
    y_max = min(cy + radius, img_height - 1)

    max_width = x_max - x_min + 1
    max_height = y_max - y_min + 1
    size = min(max_width, max_height)

    x_min = max(cx - size // 2, 0)
    y_min = max(cy - size // 2, 0)
    x_max = min(x_min + size - 1, img_width - 1)
    y_max = min(y_min + size - 1, img_height - 1)

    cropped_img = img[y_min:y_max + 1, x_min:x_max + 1]

    new_cx = (x_max - x_min) // 2
    new_cy = (y_max - y_min) // 2
    radius_effectif = min(new_cx, new_cy)

    y_coords, x_coords = np.meshgrid(np.arange(size), np.arange(size))
    distances = (x_coords - new_cx)**2 + (y_coords - new_cy)**2
    disk_mask = distances <= radius_effectif**2

    cropped_img = np.where(disk_mask[..., np.newaxis], cropped_img, 0)

    return cropped_img, [x_min, x_max, y_min, y_max]


def min_max_norm(image: np.ndarray, save_type: str = 'float32') -> np.ndarray:
    """
    Normalize image values to [0, 1] range using min-max normalization.

    Args:
        image (np.ndarray): Input image array.
        save_type (str, optional): Output array dtype. Defaults to 'float32'.

    Returns:
        np.ndarray: Normalized image array.
    """
    image = (image - image.min()) / (image.max() - image.min())
    return image.astype(save_type)


def fix_circle(img: np.ndarray, size: int = 1024, save_type: str = 'float32') -> np.ndarray:
    """
    Apply circular mask to the image, setting values outside the circle to zero.

    Creates a circular mask based on distance from center and applies it to the image.
    Used to ensure the fisheye lens area is properly isolated.

    Args:
        img (np.ndarray): Input image array.
        size (int, optional): Size of the circular mask. Defaults to 1024.
        save_type (str, optional): Output array dtype. Defaults to 'float32'.

    Returns:
        np.ndarray: Image with circular mask applied.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x, y)
    dist2ori = np.sqrt(x**2 + y**2)
    img[dist2ori > 1] = 0
    return img.astype(save_type)


def gray_contrast_energy(image: np.ndarray) -> np.ndarray:
    """
    Calculate the contrast energy map of a grayscale image.

    This function converts an RGB image to grayscale and computes its contrast energy
    using a Laplacian-like filter and non-linear mapping.

    Args:
        image (np.ndarray): Input RGB image array.

    Returns:
        np.ndarray: Normalized contrast energy map.
    """
    R, G, B = cv2.split(image)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    k = 0.1  # Contrast sensitivity parameter
    gh = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])  # Laplacian-like kernel
    I_gh = cv2.filter2D(gray, -1, gh)
    Z_c = np.sqrt(I_gh**2)
    alpha = np.max(Z_c)
    CE_c = (alpha * Z_c) / (Z_c + alpha * k) - 0.2353
    return min_max_norm(CE_c)


def texture_patch(gray_img: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Calculate local texture features for each pixel in a grayscale image.

    This function computes texture by measuring the average absolute difference
    between each pixel and its neighbors within a patch.

    Args:
        gray_img (np.ndarray): Input grayscale image array.
        patch_size (int): Size of the local neighborhood for texture calculation.

    Returns:
        np.ndarray: Array of local texture values for each pixel.
    """
    def local_texture(patch: np.ndarray) -> float:
        """Calculate texture measure for a single patch."""
        center = patch[len(patch) // 2]
        diffs = np.abs(patch - center)
        return np.sum(diffs) / (len(patch) - 1)

    return generic_filter(gray_img, local_texture, size=patch_size, mode='reflect')


def extract_rich_features(img: np.ndarray, pred_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract a comprehensive set of image features for sky segmentation.

    This function computes multiple image features including:
    - Color features (RGB, HSV)
    - Texture features (local statistics, patch-based)
    - Contrast energy
    - Uniformity and entropy measures
    - Model predictions at different scales
    - Custom color indices

    Args:
        img (np.ndarray): Input RGB image array.
        pred_dict (Dict[str, np.ndarray]): Dictionary of model predictions at different scales.

    Returns:
        np.ndarray: Array of concatenated features of shape (H*W, N_features).
    """
    kernel_size = 7  # Size of neighborhood for local feature calculation
    H, W, _ = img.shape
    img_float = img.astype(np.float32)
    img_norm = img_float / 255.0  # Normalize to [0,1] range

    gray_contrast = gray_contrast_energy(img_float)

    r = img_norm[:, :, 0].flatten()
    g = img_norm[:, :, 1].flatten()
    b = img_norm[:, :, 2].flatten()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    h = hsv_img[:, :, 0].flatten()
    s = hsv_img[:, :, 1].flatten()
    v = hsv_img[:, :, 2].flatten()

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    patch_mean = uniform_filter(gray_img, size=kernel_size, mode='reflect')
    patch_sqmean = uniform_filter(gray_img ** 2, size=kernel_size, mode='reflect')
    patch_std = np.sqrt(patch_sqmean - patch_mean ** 2)

    texture = texture_patch(gray_img, patch_size=kernel_size)
    gray_img_quant = np.clip((gray_img * 10).astype(int), 0, 9)  # 10 bins
    one_hot = np.eye(10)[gray_img_quant.reshape(-1)].reshape(H, W, 10).astype(np.float32)
    uniformity = np.zeros((H, W), dtype=np.float32)
    entropy = np.zeros((H, W), dtype=np.float32)
    for bin_idx in range(10):
        bin_map = one_hot[:, :, bin_idx]
        bin_mean = uniform_filter(bin_map, size=kernel_size, mode='reflect')
        uniformity += bin_mean ** 2
        entropy -= bin_mean * np.log2(bin_mean + 1e-9)

    return np.stack([
        pred_dict["p_sky_512"],
        pred_dict["p_sky_1024"],
        pred_dict["p_sky_2048"],
        r, g, b,
        h, v,
        patch_mean.flatten(),
        patch_std.flatten(),
        uniformity.flatten(), entropy.flatten(), texture.flatten(),
        -3.77 * r - 1.25 * g + 12.40 * b - 4.62,
        3.35 * h + 2.55 * s + 8.58 * v - 7.51,
        1.4 * b - g,
        gray_contrast.flatten()
    ], axis=1)


def model_inference(
    image_rgb: np.ndarray,
    pathModel: str,
    model_name: str = 'efficientnet-b7',
    use_lgbm: bool = False,
    resize_target: Tuple[int, int] = (1024, 1024)
) -> Optional[np.ndarray]:
    """
    Perform sky segmentation using a trained Unet++ model with optional LightGBM refinement.

    This function implements a two-stage sky segmentation approach:
    1. Primary segmentation using a Unet++ model with EfficientNet backbone
    2. Optional refinement using a LightGBM meta-model with rich feature extraction

    The function handles:
    - Model loading and validation
    - Multi-scale prediction for LightGBM
    - Feature extraction and refinement
    - Threshold application and circular mask enforcement

    Args:
        image_rgb (np.ndarray): Input RGB image array
        pathModel (str): Directory containing model checkpoints
        model_name (str): EfficientNet backbone version
        use_lgbm (bool): Whether to use LightGBM refinement
        resize_target (Tuple[int, int]): Image resolution for processing

    Returns:
        Optional[np.ndarray]: Binary sky mask (1 for sky), or None if error occurs
    """

    threshold_dict = {
        'efficientnet-b5': 176 / 255,
        'efficientnet-b7': 157 / 255
    }
    threshold = threshold_dict.get(model_name, 0.5)

    if not os.path.exists(pathModel):
        raise FileNotFoundError(f"Model folder not found at {pathModel}")

    checkpoint_path = os.path.join(pathModel, f'{model_name}.pt')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{model_name}.pt' not found in {pathModel}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")

    model = smp.UnetPlusPlus(
        encoder_name=f'{model_name}', encoder_weights='advprop',
        in_channels=3, classes=1, activation='sigmoid'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    image_resized = cv2.resize(image_rgb, resize_target, interpolation=cv2.INTER_LINEAR)

    if use_lgbm:
        lgbm_name = f"meta_model_{model_name.split('-')[-1]}.txt"
        lgbm_model_path = os.path.join(pathModel, lgbm_name)
        if not os.path.isfile(lgbm_model_path):
            raise FileNotFoundError(f"LGBM model file '{lgbm_name}' not found in {pathModel}")
        try:
            model_lgbm = lgb.Booster(model_file=lgbm_model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load LightGBM model from {lgbm_model_path}: {e}")

        scales = [512, 1024, 2048]
        pred_dict = {}
        for scale in scales:
            img_scaled = cv2.resize(image_rgb, (scale, scale), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.tensor(
                img_scaled, dtype=torch.float32).permute(
                2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                pred = model(img_tensor).squeeze().cpu().numpy()
            pred_resized = cv2.resize(pred, resize_target, interpolation=cv2.INTER_LINEAR)
            pred_dict[f"p_sky_{scale}"] = pred_resized.flatten()

        X_lgbm = extract_rich_features(image_resized, pred_dict)
        pred_lgbm = model_lgbm.predict(X_lgbm)
        pred_mask = pred_lgbm.reshape(*resize_target)
    else:
        img_tensor = torch.tensor(
            image_resized, dtype=torch.float32).permute(
            2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            pred = model(img_tensor).squeeze().cpu().numpy()
        pred_mask = cv2.resize(pred, resize_target, interpolation=cv2.INTER_LINEAR)

    pred_mask = (pred_mask > threshold).astype(np.uint8)
    pred_mask = 1 - pred_mask
    pred_mask = fix_circle(pred_mask, size=resize_target[0])
    return pred_mask


def inference(
    img: np.ndarray,
    model_name: str = 'efficientnet-b5',
    use_lgbm: bool = False,
    resize_target: Tuple[int, int] = (1024, 1024)
) -> Optional[np.ndarray]:
    """
    Process a single fisheye image to create a sky segmentation mask.

    This function performs sky segmentation on a single image using the following steps:
    1. Estimates the fisheye lens parameters from camera calibration
    2. Crops the image around the fisheye disk region
    3. Applies the trained segmentation model
    4. Post-processes and resizes results to match original image dimensions

    Args:
        img: Input RGB image array as numpy array
        model_name: Name of the EfficientNet model architecture ('efficientnet-b5' or 'efficientnet-b7')
        use_lgbm: Whether to use LightGBM meta-model for refinement (improves accuracy but slower)
        resize_target: Target image size for model processing (width, height)

    Returns:
        Binary mask where 1 indicates sky pixels, 0 indicates obstacles/non-sky

    Note:
        Requires camera calibration data to be available and model weights in SystemData folder
    """
    data_dir = "./DebugData"
    pprad_path = os.path.join(data_dir, "pprad.yml")
    estimate_radius(pprad_path)
    img_cropped, cropped_section = crop_around_disk(pprad_path, img)
    im_height, im_width, _ = img.shape
    cv2.imwrite(
        os.path.join(
            data_dir, "img_cropped.jpg"), cv2.cvtColor(
            img_cropped, cv2.COLOR_BGR2RGB))
    prediction = model_inference(
        img_cropped,
        "./SystemData",
        model_name=model_name,
        use_lgbm=use_lgbm,
        resize_target=resize_target)
    
    cv2.imwrite(os.path.join(data_dir, "model_output.jpg"), prediction * 255)
    size = cropped_section[3] + 1 - cropped_section[2]
    prediction = cv2.resize(prediction, (size, size), interpolation=cv2.INTER_NEAREST)
    full_size_mask = np.zeros((im_height, im_width), dtype=np.uint8)
    full_size_mask[cropped_section[2]:cropped_section[3] + 1,
                   cropped_section[0]:cropped_section[1] + 1] = prediction
    return full_size_mask


def batch_disk_mask_inference(
    folder_path: str = "./SkyImageOfSite/",
    model_path: str = "./SystemData",
    model_name: str = 'efficientnet-b5',
    use_lgbm: bool = False,
    resize_target: Tuple[int, int] = (1024, 1024)
) -> Optional[np.ndarray]:
    """
    Process multiple fisheye images and create a combined mask.

    This function processes all images in a specified folder using the sky segmentation model
    and combines their masks using an AND operation. The combined mask represents areas
    that are consistently classified as sky across all images.

    Args:
        folder_path (str): Path to folder containing input images.
        model_path (str): Path to folder containing model checkpoints.
        model_name (str): Name of the model architecture to use.
        use_lgbm (bool): Whether to use LightGBM for post-processing.
        resize_target (Tuple[int, int]): Target size for image resizing.

    Returns:
        np.ndarray: Combined binary mask.
    """
    import glob
    data_dir = "./DebugData"
    pprad_path = os.path.join(data_dir, "pprad.yml")
    estimate_radius(pprad_path)

    image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
        glob.glob(os.path.join(folder_path, "*.png"))
    combined_mask = None

    for _, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cropped, cropped_section = crop_around_disk(pprad_path, img)
        prediction = model_inference(
            img_cropped,
            model_path,
            model_name=model_name,
            use_lgbm=use_lgbm,
            resize_target=resize_target)
        
        size = cropped_section[3] + 1 - cropped_section[2]
        prediction = cv2.resize(prediction, (size, size), interpolation=cv2.INTER_NEAREST)

        full_size_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        full_size_mask[cropped_section[2]:cropped_section[3] + 1,
                       cropped_section[0]:cropped_section[1] + 1] = prediction

        if combined_mask is None:
            combined_mask = full_size_mask.copy()
        else:
            combined_mask = combined_mask * full_size_mask  # set to 0 if any mask is 0

    cv2.imwrite(os.path.join(data_dir, "combined_mask.jpg"), combined_mask * 255)
    return combined_mask
