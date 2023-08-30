import ants
import numpy as np
from antspynet import brain_extraction
from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize import zscore, fcm, kde, whitestripe


def compute_histogram(image, bins=256):
    hist, _ = np.histogram(image, bins=bins, range=(image.min(), image.max()))
    return hist


def add_padding(img, desired_dim, axis=1):
    """
    Add padding to an ANTs image in the specified dimension until the desired dimension is reached.

    Args:
    - img (ants.ANTsImage): Input image to be padded.
    - desired_dim (int): The desired size for the specified dimension.
    - axis (int): The axis where padding should be added. Default is 1.

    Returns:
    - ants.ANTsImage: Padded image.
    """

    # Calculate how much padding is needed
    current_dim = img.shape[axis]
    padding_needed = desired_dim - current_dim

    if padding_needed <= 0:
        return img

    # Calculate padding for both sides
    before_padding = padding_needed // 2
    after_padding = padding_needed - before_padding

    # Add padding based on the specified axis
    if axis == 0:
        padded_array = np.pad(img.numpy(), ((before_padding, after_padding), (0, 0), (0, 0)))
    elif axis == 1:
        padded_array = np.pad(img.numpy(), ((0, 0), (before_padding, after_padding), (0, 0)))
    elif axis == 2:
        padded_array = np.pad(img.numpy(), ((0, 0), (0, 0), (before_padding, after_padding)))
    else:
        raise ValueError("Invalid axis value. It should be 0, 1, or 2.")

    # Convert back to ANTsImage
    padded_image = ants.from_numpy(padded_array, origin=img.origin, spacing=img.spacing, direction=img.direction)
    return padded_image


def spatial_normalization(nifti_img, target_size, target_spacing, target_origin, target_direction, interp_type=3):
    resampled_img = ants.resample_image(nifti_img, target_size, use_voxels=True, interp_type=interp_type)  # Reshape
    resampled_img.set_spacing(target_spacing)  # Set spacing
    resampled_img = ants.reorient_image2(resampled_img, 'RAI')  # Reorient to RAI
    resampled_img.set_origin(target_origin)  # Set origin
    resampled_img.set_direction(target_direction)  # Set direction
    return resampled_img


def truncate_intensities(image, quantile_values=(0.01, 0.99)):
    preprocessed_image = ants.image_clone(image)
    quantiles = (image.quantile(quantile_values[0]), image.quantile(quantile_values[1]))
    preprocessed_image[image < quantiles[0]] = quantiles[0]
    preprocessed_image[image > quantiles[1]] = quantiles[1]
    return preprocessed_image


def extract_brain(image, brain_extraction_modality):
    preprocessed_image = ants.image_clone(image)
    probability_mask = brain_extraction(
        preprocessed_image,
        modality=brain_extraction_modality
    )
    mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
    mask = ants.morphology(mask, "close", 6).iMath_fill_holes()
    return mask


def segment_image(image, mask, **kwargs):
    return ants.mask_image(image, mask, **kwargs)


def bias_field_correction(image, mask=None, shrink_factor=4, return_bias_field=False, **kwargs):
    preprocessed_image = ants.image_clone(image)
    return ants.n4_bias_field_correction(
        preprocessed_image, mask, shrink_factor=shrink_factor, return_bias_field=return_bias_field, **kwargs
    )


def denoise_image(image, mask=None, shrink_factor=1):
    preprocessed_image = ants.image_clone(image)
    if mask is None:
        return ants.denoise_image(preprocessed_image, shrink_factor=shrink_factor)
    return ants.denoise_image(preprocessed_image, mask, shrink_factor=shrink_factor)


def match_intensities(image, reference_image):
    preprocessed_image = ants.image_clone(image)
    return ants.histogram_match_image(preprocessed_image, reference_image)


def interpolate_value_distribution(value, ref_list, target_list):
    from scipy.stats import percentileofscore

    sorted_ref = sorted(ref_list)

    def interpolate(value_):  # Function to find the corresponding value in the reference list
        quantile = percentileofscore(target_list, value_) / 100.0
        return np.percentile(sorted_ref, quantile * 100)

    return interpolate(value)


def normalize_intensity(image, modality, mask=None, normalization_type='01'):
    preprocessed_image = ants.image_clone(image)

    if modality == 'T1w' or modality == 'T1w_ce':
        modality_ = Modality.T1
    elif modality == 'T2w':
        modality_ = Modality.T2
    elif modality == 'FLAIR':
        modality_ = Modality.FLAIR
    else:
        modality_ = None

    if normalization_type == '01':
        return (preprocessed_image - preprocessed_image.min()) / (preprocessed_image.max() - preprocessed_image.min())

    elif normalization_type == '0mean':
        return (preprocessed_image - preprocessed_image.mean()) / preprocessed_image.std()

    elif normalization_type == 'ZScore':
        nib_img = ants.to_nibabel(preprocessed_image).get_fdata()
        zscore_norm = zscore.ZScoreNormalize()
        norm_img = zscore_norm(nib_img, mask=mask, modality=modality_)
        return ants.from_numpy(norm_img)

    elif normalization_type == 'FCM':  # T1w + c?
        nib_img = ants.to_nibabel(preprocessed_image).get_fdata()
        fcm_norm = fcm.FCMNormalize(tissue_type=TissueType.WM)
        norm_img = fcm_norm(nib_img, mask=mask, modality=modality_)
        return ants.from_numpy(norm_img)

    elif normalization_type == 'KDE':  # FLAIR?
        nib_img = ants.to_nibabel(preprocessed_image).get_fdata()
        kde_norm = kde.KDENormalize()
        norm_img = kde_norm(nib_img, mask=mask, modality=modality_)
        return ants.from_numpy(norm_img)

    elif normalization_type == 'WhiteStripe':  # T1w?
        nib_img = ants.to_nibabel(preprocessed_image).get_fdata()
        ws_norm = whitestripe.WhiteStripeNormalize()
        norm_img = ws_norm(nib_img, mask=mask, modality=modality_)
        return ants.from_numpy(norm_img)

    else:
        raise ValueError("Unrecognized Intensity Normalization type.")
