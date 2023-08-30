import ants
from preprocessing.utils import get_modality
from preprocessing.functions import normalize_intensity
from antspynet.utilities import brain_extraction, regression_match_image, get_antsxnet_data


def preprocess_brain_images(
        image_t1w,
        image_t1w_ce,
        image_t2w,
        image_flair,
        reference_modality='T1w_ce',
        truncate_intensity=(0.01, 0.99),
        do_brain_extraction=True,
        template_transform_type=None,
        template=None,  # "biobank",
        do_bias_correction=True,
        return_bias_field=False,
        do_denoising=True,
        intensity_matching_type=None,
        reference_image=None,
        intensity_normalization_type=None,
        antsxnet_cache_directory=None,
        verbose=True
):
    preprocessed_images = {
        'T1w': ants.image_clone(image_t1w),
        'T1w_ce': ants.image_clone(image_t1w_ce),
        'T2w': ants.image_clone(image_t2w),
        'FLAIR': ants.image_clone(image_flair),
    }
    assert reference_modality in preprocessed_images.keys()

    # Truncate intensity
    if truncate_intensity is not None:
        for modality, image in preprocessed_images.items():
            quantiles = (image.quantile(truncate_intensity[0]), image.quantile(truncate_intensity[1]))
            if verbose:
                print(
                    f"Preprocessing: Truncate intensities ({modality}) image (low = {quantiles[0]:.6f},  high = {quantiles[1]:.6f}).")

            preprocessed_images[modality][image < quantiles[0]] = quantiles[0]
            preprocessed_images[modality][image > quantiles[1]] = quantiles[1]

    # Brain extraction
    masks = {}
    if do_brain_extraction:
        for modality, preprocessed_image in preprocessed_images.items():
            if verbose:
                print(f"\nPreprocessing:  brain extraction for {modality} image.")
            probability_mask = brain_extraction(
                preprocessed_image,
                modality=get_modality(modality),
                antsxnet_cache_directory=antsxnet_cache_directory,
                verbose=verbose
            )
            mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
            mask = ants.morphology(mask, "close", 6).iMath_fill_holes()
            masks[modality] = mask

    # Template normalization
    transforms = {}
    if template_transform_type is not None:
        template_image = None
        if template is None:
            template_image = preprocessed_images[reference_modality]
        elif isinstance(template, str):
            template_file_name_path = get_antsxnet_data(template, antsxnet_cache_directory=antsxnet_cache_directory)
            template_image = ants.image_read(template_file_name_path)
        else:
            template_image = template

        if not masks:
            for modality, preprocessed_image in preprocessed_images.items():
                if modality == reference_modality:
                    continue
                registration = ants.registration(
                    fixed=template_image,
                    moving=preprocessed_image,
                    type_of_transform=template_transform_type,
                    verbose=True if verbose > 1 else False
                )
                preprocessed_images[modality] = registration['warpedmovout']
                transforms[modality] = dict(fwdtransforms=registration['fwdtransforms'],
                                            invtransforms=registration['invtransforms'])
        else:

            template_file_name_path = get_antsxnet_data('biobank', antsxnet_cache_directory=antsxnet_cache_directory)
            temp_template_img = ants.image_read(template_file_name_path)
            registration = ants.registration(
                fixed=temp_template_img,  # biobank reference
                moving=preprocessed_images[reference_modality],  # T1w_ce
                type_of_transform='Rigid',  # Rigid / Affine / SyN
                verbose=True if verbose > 1 else False
            )
            preprocessed_images[reference_modality] = registration['warpedmovout']

            template_image = preprocessed_images[reference_modality]  # ?? could be commented
            template_probability_mask = brain_extraction(
                template_image,
                modality=get_modality(reference_modality),
                antsxnet_cache_directory=antsxnet_cache_directory,
                verbose=True if verbose > 1 else False
            )
            template_mask = ants.threshold_image(template_probability_mask, 0.5, 1, 1, 0)
            masks[reference_modality] = template_mask
            template_brain_image = template_mask * template_image

            for modality, preprocessed_image in preprocessed_images.items():
                if modality == reference_modality:
                    continue

                preprocessed_brain_image = preprocessed_image * masks[modality]
                registration = ants.registration(
                    fixed=template_brain_image,
                    moving=preprocessed_brain_image,
                    type_of_transform=template_transform_type,
                    verbose=True if verbose > 1 else False
                )
                transforms[modality] = dict(fwdtransforms=registration['fwdtransforms'],
                                            invtransforms=registration['invtransforms'])

                preprocessed_images[modality] = ants.apply_transforms(
                    fixed=preprocessed_images[reference_modality],
                    moving=preprocessed_image,
                    transformlist=registration['fwdtransforms'], interpolator="linear",
                    verbose=True if verbose > 1 else False
                )
                masks[modality] = ants.apply_transforms(
                    fixed=preprocessed_images[reference_modality],
                    moving=masks[modality],
                    transformlist=registration['fwdtransforms'],
                    interpolator="genericLabel",
                    verbose=True if verbose > 1 else False
                )

    # Do bias correction
    bias_fields = {}
    if do_bias_correction:
        for modality, preprocessed_image in preprocessed_images.items():
            if verbose:
                print(f"\nPreprocessing:  brain correction for {modality} image.")

            mask = masks.get(modality)
            n4_output = ants.n4_bias_field_correction(
                preprocessed_image, mask=mask, shrink_factor=4,
                return_bias_field=return_bias_field, verbose=verbose
            )
            if return_bias_field:
                bias_fields[modality] = n4_output
            else:
                preprocessed_images[modality] = n4_output

    # Denoising
    if do_denoising:
        for modality, preprocessed_image in preprocessed_images.items():
            if verbose:
                print(f"\nPreprocessing:  denoising {modality} image.")

            mask = masks[modality]
            if mask is None:
                preprocessed_images[modality] = ants.denoise_image(preprocessed_image, shrink_factor=1)
            else:
                preprocessed_images[modality] = ants.denoise_image(preprocessed_image, mask, shrink_factor=1)

    # Image matching
    if reference_image is not None and intensity_matching_type is not None:
        for modality, preprocessed_image in preprocessed_images.items():
            if verbose:
                print(f"\nPreprocessing:  intensity matching for {modality} image.")

            if intensity_matching_type == "regression":
                preprocessed_images[modality] = regression_match_image(preprocessed_image, reference_image)
            elif intensity_matching_type == "histogram":
                preprocessed_images[modality] = ants.histogram_match_image(preprocessed_image, reference_image)
            else:
                raise ValueError("Unrecognized intensity_matching_type.")

    # Intensity normalization
    if intensity_normalization_type is not None:
        for modality, preprocessed_image in preprocessed_images.items():
            if verbose:
                print(f"Preprocessing:  intensity normalization for {modality} image.")
            preprocessed_images[modality] = normalize_intensity(preprocessed_image, modality, masks[modality],
                                                                intensity_normalization_type)

    return_dict = {
        'T1w': {'preprocessed_image': preprocessed_images['T1w']},
        'T1w_ce': {'preprocessed_image': preprocessed_images['T1w_ce']},
        'T2w': {'preprocessed_image': preprocessed_images['T2w']},
        'FLAIR': {'preprocessed_image': preprocessed_images['FLAIR']},
    }

    for modality in return_dict.keys():
        if masks and masks.get(modality):
            return_dict[modality]['brain_mask'] = masks[modality]
        elif bias_fields and bias_fields.get(modality):
            return_dict[modality]['bias_field'] = bias_fields[modality]
        elif transforms and transforms.get(modality):
            return_dict[modality]['template_transforms'] = transforms[modality]

    return return_dict
