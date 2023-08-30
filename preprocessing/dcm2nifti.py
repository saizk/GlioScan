import os
from typing import List, Type
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


@dataclass
class NiftiSettings:
    gen_bids_file: bool = True
    bids_anonymization: bool = True
    merge_2d_slice: bool = False
    philips_precise_scaling: bool = False
    verbose: bool = False
    crop_output: bool = False
    compression: bool = False

    @classmethod
    def from_dict(cls: Type['NiftiSettings'], obj: dict):
        return cls(**obj)


class Dicom2Nifti:
    _BASE_CMD = 'dcm2niix'

    def __init__(
            self,
            gen_bids_file: bool = True,
            bids_anonymization: bool = False,
            merge_2d_slice: bool = False,
            philips_precise_scaling: bool = False,
            verbose: bool = False,
            crop_output: bool = False,
            compression: bool = False
    ):
        self.settings = self._parse_params(locals())

    @staticmethod
    def _parse_params(params):
        del params['self']
        return NiftiSettings.from_dict(params)

    def _gen_command(self):
        cmd = self._BASE_CMD
        cmd += ' -b y' if self.settings.gen_bids_file else ''
        cmd += ' -ba y' if self.settings.bids_anonymization else ''
        cmd += ' -m y' if self.settings.merge_2d_slice else ''
        cmd += ' -p y' if self.settings.philips_precise_scaling else ''
        cmd += ' -v 1' if self.settings.verbose else ''
        cmd += ' -x y' if self.settings.crop_output else ''
        cmd += ' -z y' if self.settings.compression else ''
        cmd += ' -f %f'
        return cmd

    def gen_command(self, input_directory: str, output_directory: str = None):
        cmd = self._gen_command()
        cmd += f' -o "{output_directory or input_directory}" "{input_directory}"'
        return cmd

    def gen_commands(self, i_o_directories: list):
        for input_dir, out_dir in i_o_directories:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            yield self.gen_command(input_dir, out_dir)

    def convert_directory(self, input_directory: str, output_directory: str):
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        cmd = self.gen_command(input_directory, output_directory)
        os.system(cmd)

    def parallel_convert_directories(self, i_o_directories: list):
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            commands = list(self.gen_commands(i_o_directories))
            results = pool.map(os.system, commands)


def dcm2nifti(input_dir: str):
    def name_changer(path):
        parent = Path(path)
        return parent.parent
    parser = Dicom2Nifti()
    input_dirs = list(dirpath for dirpath, dirnames, filenames in os.walk(input_dir) if not dirnames)
    # pprint(input_dirs)
    output_dirs = list(map(lambda x: name_changer(x), input_dirs))
    # pprint(output_dirs)
    i_o_directories = list(zip(input_dirs, output_dirs))
    parser.parallel_convert_directories(i_o_directories)
    # parser.convert_directory(test, test)
