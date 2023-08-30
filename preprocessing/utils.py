import os
import json
from pathlib import Path

import ants
import numpy as np
import nibabel as nib

from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_ = json.load(f)
    return json_


def load_file(root, scan, read_with='ants'):
    if scan == 'T1w':
        root += '_T1w'
    elif scan == 'T1w_ce':
        root += '_ce-GADOLINIUM_T1w'
    elif scan == 'T2w':
        root += '_T2w'
    elif scan == 'FLAIR':
        root += '_FLAIR'

    if read_with == 'ants':
        return ants.image_read(f'{root}.nii.gz')
    elif read_with == 'nibabel':
        return nib.load(f'{root}.nii.gz')


def load_gm_file(root, patient_no, scan, read_with='ants'):
    folder = f'{root}/sub-{patient_no:04}/anat/sub-{patient_no:04}'
    return load_file(folder, scan, read_with)


def load_ucsf_file(root, patient_no, scan, read_with='ants'):
    folder = f'{root}/sub-{patient_no:04}/anat/sub-{patient_no:04}'
    return load_file(folder, scan, read_with)


def load_mask(root, patient_no, mask_type='tumor', read_with='ants'):
    path = f'{root}/derivatives/sub-{patient_no:04}'
    if mask_type == 'brain':
        path += f'/sub-{patient_no:04}_desc-brain_mask.nii.gz'
    elif mask_type == 'tumor':
        path += f'/sub-{patient_no:04}_desc-tumor_dseg.nii.gz'
    else:
        raise ValueError(f'Unknown mask type: {mask_type}')

    if read_with == 'ants':
        return ants.image_read(path).numpy()
    elif read_with == 'nibabel':
        return nib.load(path).get_fdata()
    else:
        raise ValueError(f'Unknown read method: {read_with}')


def load_nifti(path, read_with='ants'):
    if read_with == 'ants':
        return ants.image_read(path).numpy()
    elif read_with == 'nibabel':
        return nib.load(path).get_fdata()
    else:
        raise ValueError(f'Unknown read method: {read_with}')


def get_patient_id(patient):
    return patient.split('_')[0]


def get_patient_no(patient):
    id_ = get_patient_id(patient)
    number = id_.split('-')[-1]
    return int(number)


def get_modality(modality):
    if 'T1w' in modality:  # T1w / T1w + contrast
        return 't1'
    elif 'T2w' in modality:  # T2w
        return 't2'
    elif 'FLAIR' in modality:  # FLAIR
        return 'flair'


def get_mask_name(filename, modality_in_name=True):
    if 'desc' in filename and 'mask' in filename:
        return filename
    if not modality_in_name:
        id_ = get_patient_id(filename)
        modality = 'brain'
    else:
        id_, *modality = filename.split('_')
        modality = '_'.join(modality)[:-7]
    return f'{id_}_desc-{modality}_mask.nii.gz'


def get_mask_path(root_directory, filename, **kwargs):
    mask_root = Path(root_directory) / 'derivatives' / get_patient_id(filename)
    mask_root.mkdir(parents=True, exist_ok=True)
    mask_path = str(mask_root / get_mask_name(filename, **kwargs))
    return mask_path


def get_paths(root_directory, ommit_derivatives=False, format_='nii.gz'):
    results = {'T1w': [], 'T1w_ce': [], 'T2w': [], 'FLAIR': []}

    for dirpath, dirs, files in os.walk(root_directory):  # Iterate over root_directory
        if 'derivatives' in dirpath and ommit_derivatives:
            continue
        for filename in files:  # For each file
            if not filename.endswith(format_):
                continue
            if filename.endswith(f'ce-GADOLINIUM_T1w.{format_}'):
                results['T1w_ce'].append(os.path.join(dirpath, filename))
            elif filename.endswith(f'T1w.{format_}'):
                results['T1w'].append(os.path.join(dirpath, filename))
            elif filename.endswith(f'T2w.{format_}'):
                results['T2w'].append(os.path.join(dirpath, filename))
            elif filename.endswith(f'FLAIR.{format_}'):
                results['FLAIR'].append(os.path.join(dirpath, filename))
    return results


def get_nifti_paths(root_directory, ommit_derivatives=False):
    return get_paths(root_directory, ommit_derivatives=ommit_derivatives, format_='nii.gz')


def get_npy_paths(root_directory, ommit_derivatives=False):
    return get_paths(root_directory, ommit_derivatives=ommit_derivatives, format_='npy')


def get_npz_paths(root_directory, ommit_derivatives=False):
    return get_paths(root_directory, ommit_derivatives=ommit_derivatives, format_='npz')


@lru_cache()
def load_images(root_directory, modality='T1w'):
    images = []
    for dirpath, dirs, files in os.walk(root_directory):  # Iterate over root_directory
        for filename in files:  # For each file
            if filename.endswith('ce-GADOLINIUM_T1w.nii.gz') and modality == 'T1w_ce':
                images.append(os.path.join(dirpath, filename))
            elif filename.endswith('T1w.nii.gz') and modality == 'T1w':
                images.append(os.path.join(dirpath, filename))
            elif filename.endswith('T2w.nii.gz') and modality == 'T2w':
                images.append(os.path.join(dirpath, filename))
            elif filename.endswith('FLAIR.nii.gz') and modality == 'FLAIR':
                images.append(os.path.join(dirpath, filename))

    with ProcessPoolExecutor() as pool:
        results = list(pool.map(nib.load, images))

    return [nifti.get_fdata() for nifti in results]


def process_file(args):
    filename, dirpath, data_type = args
    pat_id = filename.split('_')[0]
    image = ants.image_read(os.path.join(dirpath, filename)).numpy()
    return data_type, pat_id, {
        'Mean': float(np.mean(image)),
        'Var': float(np.var(image)),
        'Min': float(np.min(image)),
        'Max': float(np.max(image)),
        'Sum': float(np.sum(image)),
    }


def gen_metadata(root_directory, json_path):
    data = {'T1w': {}, 'T1w_ce': {}, 'T2w': {}, 'FLAIR': {}}
    tasks = []

    for dirpath, dirs, files in os.walk(root_directory):  # Iterate over root_directory
        for filename in files:  # For each file
            if filename.endswith('GADOLINIUM_T1w.nii.gz'):
                tasks.append((filename, dirpath, 'T1w_ce'))
            elif filename.endswith('T1w.nii.gz'):
                tasks.append((filename, dirpath, 'T1w'))
            elif filename.endswith('T2w.nii.gz'):
                tasks.append((filename, dirpath, 'T2w'))
            elif filename.endswith('FLAIR.nii.gz'):
                tasks.append((filename, dirpath, 'FLAIR'))

    with ProcessPoolExecutor() as executor:
        for data_type, pat_id, result in executor.map(process_file, tasks):
            data[data_type][pat_id] = result

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def show_image_info(image):
    if isinstance(image, ants.core.ants_image.ANTsImage):
        image_array = image.numpy()
    elif isinstance(image, nib.nifti1.Nifti1Image):
        image_array = image.get_fdata()
    elif isinstance(image, np.ndarray):
        image_array = image
    elif isinstance(image, str) and image.endswith('.npy'):
        image_array = np.load(image)
    elif isinstance(image, str) and image.endswith('.nii.gz'):
        image_array = ants.image_read(image).numpy()
    else:
        raise ValueError(f'Unknown image type: {type(image)}')

    print(f'Mean: {np.mean(image_array):.2f}')
    print(f'Variance: {np.var(image_array):.2f}')
    print(f'Min: {np.min(image_array):.2f}')
    print(f'Max: {np.max(image_array):.2f}')
    return np.mean(image_array), np.var(image_array), np.min(image_array), np.max(image_array)
