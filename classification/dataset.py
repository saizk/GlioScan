import numpy as np
from monai.data import ImageDataset
from monai.transforms import Randomizable, apply_transform


class EnhancedImageDataset(ImageDataset):
    def __init__(
            self,
            image_files, labels, feature_vector,
            spatial_transformation=None,
            intensity_transformation=None,
            verbose=False,
            **kwargs
    ):
        super().__init__(image_files=image_files, labels=labels, image_only=True, **kwargs)
        self.feature_vector = feature_vector
        self.spatial_transformation = spatial_transformation
        self.intensity_transformation = intensity_transformation
        self.verbose = verbose

        if isinstance(self.transform, Randomizable):
            self.transform.set_random_state(seed=self._seed)
        if isinstance(self.spatial_transformation, Randomizable):
            self.spatial_transformation.set_random_state(seed=self._seed)
        if isinstance(self.intensity_transformation, Randomizable):
            self.intensity_transformation.set_random_state(seed=self._seed)

    def __getitem__(self, index: int):
        self.randomize()

        # load data
        sequences = [self.loader(sequence) for sequence in self.image_files[index]]
        features = self.feature_vector[index]
        label = self.labels[index]

        if self.transform is None:
            sequences = [seq.reshape(1, *seq.shape) for seq in sequences]
            if self.spatial_transformation is not None:
                # Apply spatial transformations and check shape
                sequences = [apply_transform(self.spatial_transformation, seq, map_items=False) for seq in sequences]

            if self.intensity_transformation is not None:
                # Apply intensity transformations and check shape
                for transform in self.intensity_transformation:
                    sequences = [apply_transform(transform, seq, map_items=False) for seq in sequences]

            img = np.vstack(sequences) if len(sequences) > 1 else sequences[0]

        if self.transform is not None:
            img = np.stack(sequences, axis=0)
            img = apply_transform(self.transform, img, map_items=False)

        if self.verbose:
            return img, features, label, self.image_files[index][0].split('/')[-1].split('.')[0].split('_')[0]
        return img, features, label
