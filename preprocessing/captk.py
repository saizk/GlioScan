import os
import pathlib
import shutil
from pathlib import Path


class CaPTk:

    CaPTk_PATH = 'C:/CaPTk_Full/1.9.0'

    def __init__(self, bids_folder: str, output_folder: str or pathlib.Path = None):
        self.bids_folder = bids_folder
        self.output_folder = output_folder
        self.paths = self._get_paths()

    def _get_paths(self):
        paths = {}
        for dirpath, dirs, files in os.walk(self.bids_folder):  # Iterate over root_directory
            if 'derivatives' in dirpath:
                continue
            for filename in files:  # For each file
                patient_id = filename.split('_')[0]
                if filename.endswith('_ce-GADOLINIUM_T1w.nii.gz'):
                    mr_path = {'t1c': os.path.join(dirpath, filename)}
                elif filename.endswith('_T1w.nii.gz'):
                    mr_path = {'t1': os.path.join(dirpath, filename)}
                elif filename.endswith('_T2w.nii.gz'):
                    mr_path = {'t2': os.path.join(dirpath, filename)}
                elif filename.endswith('_FLAIR.nii.gz'):
                    mr_path = {'flair': os.path.join(dirpath, filename)}
                else:
                    continue
                if patient_id not in paths:
                    paths[patient_id] = mr_path
                else:
                    paths.get(patient_id).update(mr_path)
                # paths[patient_id] = mr_path if patient_id not in paths else paths.get(patient_id).update(mr_path)

        return paths

    def run_brats_pipeline(self):
        for i, (patient, mr_paths) in enumerate(self.paths.items()):
            print(f'Running BraTS Pre-Processing Pipeline for patient {patient}...')
            t1_path = mr_paths.get('t1')
            t1c_path = mr_paths.get('t1c')
            t2_path = mr_paths.get('t2')
            flair_path = mr_paths.get('flair')
            output_dir = self.output_folder if self.output_folder is not None else Path(t1c_path).parent
            self._run_brats_pipeline(t1_path, t1c_path, t2_path, flair_path, str(output_dir))
            self.clean_directory(output_dir)

    def run_tumor_segmentation(self):
        for i, (patient, mr_paths) in enumerate(self.paths.items()):
            print(f'Running BraTS Pre-Processing Pipeline for patient {patient}...')
            t1_path = mr_paths.get('t1')
            t1c_path = mr_paths.get('t1c')
            t2_path = mr_paths.get('t2')
            flair_path = mr_paths.get('flair')
            output_folder = self.output_folder / patient
            output_folder.mkdir(parents=True, exist_ok=True)
            output_file = output_folder / f'{patient}_desc-tumor_dseg.nii.gz'
            brain_mask = output_folder / f'{patient}_desc-brain_mask.nii.gz'
            self._run_deepmedic(t1_path, t1c_path, t2_path, flair_path, None, str(output_file))

    def _run_brats_pipeline(self, t1_path: str, t1c_path: str, t2_path: str, flair_path: str, output_dir: str):
        os.system(
            f'{self.CaPTk_PATH}/bin/BraTSPipeline.exe' +
            f' -t1 {t1_path}' +
            f' -t1c {t1c_path}' +
            f' -t2 {t2_path}' +
            f' -fl {flair_path}' +
            f' -o {output_dir}'
        )

    def _run_deepmedic(self, t1_path: str, t1c_path: str, t2_path: str, flair_path: str,
                       brain_mask: str or None, output_file: str):
        cmd = f'{self.CaPTk_PATH}/bin/DeepMedic.exe' + \
            f' -md {self.CaPTk_PATH}/data/deepMedic/saved_models/brainTumorSegmentation' + \
            f' -i {t1_path},{t1c_path},{t2_path},{flair_path}'
        cmd += f' -m {brain_mask}' if brain_mask is not None else ''
        cmd += f' -o {output_file}'
        os.system(cmd)

    @staticmethod
    def clean_directory(folder):
        path = Path(folder)
        new_file = None
        for file in path.iterdir():
            if not str(file).endswith('SRI_brain.nii.gz') and not str(file).endswith('Mask_SRI.nii.gz'):
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    os.remove(file)
        for file in path.iterdir():
            if str(file).endswith('T1_to_SRI_brain.nii.gz'):
                new_file = file.parent / f'{file.parent.parent.stem}_T1w.nii.gz'
            elif str(file).endswith('T1CE_to_SRI_brain.nii.gz'):
                new_file = file.parent / f'{file.parent.parent.stem}_ce-GADOLINIUM_T1w.nii.gz'
            elif str(file).endswith('T2_to_SRI_brain.nii.gz'):
                new_file = file.parent / f'{file.parent.parent.stem}_T2w.nii.gz'
            elif str(file).endswith('FL_to_SRI_brain.nii.gz'):
                new_file = file.parent / f'{file.parent.parent.stem}_FLAIR.nii.gz'
            elif str(file).endswith('brainTumorMask_SRI.nii.gz'):
                new_file = file.parent / f'{file.parent.parent.stem}_desc-tumor_dseg.nii.gz'
            elif str(file).endswith('brainMask_SRI.nii.gz'):
                new_file = file.parent / f'{file.parent.parent.stem}_desc-brain_mask.nii.gz'

            if new_file is not None:
                os.rename(file, new_file)


def captk_run(nifti_directory: str, output_directory: str = None):
    output_directory = Path(nifti_directory) / 'derivatives'
    captk = CaPTk(nifti_directory, output_directory)
    # pprint(captk.paths)
    # captk.run_brats_pipeline()
    captk.run_tumor_segmentation()
    # captk.clean_directory(nifti_directory)

