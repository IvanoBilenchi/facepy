import cv2.cv2 as cv2
import numpy as np
import os
import sys
from os import path
from typing import Callable, Iterable, Optional


class DataSample:

    @property
    def name(self) -> str:
        components = path.basename(self.file_path).split('_')
        return ' '.join(components[:-1])

    @property
    def dir_path(self) -> str:
        return path.dirname(self.file_path)

    @property
    def image(self) -> np.array:
        if self.__image is None:
            self.__image = cv2.imread(self.file_path)
        return self.__image

    @image.setter
    def image(self, image: np.array) -> None:
        self.__image = image

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.__image: np.array = None


class Dataset:

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    def negative_verification_images(self, person_name: str = None,
                                     sample_filter: Callable = None,
                                     max_samples: int = sys.maxsize) -> Iterable[np.array]:

        skip_person: str = None
        n_samples = 0

        for sample in self.samples(sample_filter):

            sample_name = sample.name

            if skip_person != sample_name:
                skip_person = None

            if sample_name == person_name or sample_name == skip_person:
                continue

            skip_person = sample_name

            if n_samples < max_samples:
                n_samples += 1
                yield sample.image
            else:
                break

    def samples_in_dir(self, dir_path: str, sample_filter: Callable = None) -> Iterable[DataSample]:
        paths = [path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]
        paths.sort()

        for file_path in paths:
            sample = DataSample(file_path)

            if sample_filter:
                sample = sample_filter(sample)

            if sample:
                yield sample

    def samples_for_person(self, person_name: str,
                           sample_filter: Callable = None) -> Iterable[DataSample]:
        person_dir = path.join(self.data_dir, person_name.replace(' ', '_'))
        return self.samples_in_dir(person_dir, sample_filter=sample_filter)

    def samples(self, sample_filter: Callable[[DataSample],
                                              Optional[DataSample]] = None) -> Iterable[DataSample]:
        dirs = [path.join(self.data_dir, d) for d in os.listdir(self.data_dir)]
        dirs = [d for d in dirs if path.isdir(d)]
        dirs.sort()

        for dir_path in dirs:
            for sample in self.samples_in_dir(dir_path, sample_filter=sample_filter):
                yield sample
