import cv2.cv2 as cv2
import numpy as np
import os
import sys
from os import path
from typing import Callable, Iterable

from facepy import config


class DataSample:

    @property
    def person_name(self) -> str:
        return person_name_from_dir(self.dir_path)

    @property
    def file_name(self) -> str:
        return path.basename(self.file_path)

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


def person_name_from_dir(dir_path: str) -> str:
    return path.basename(dir_path).replace('_', ' ')


def all_dirs() -> Iterable[str]:
    data_dir = config.Paths.DATASET_DIR
    dirs = [path.join(data_dir, d) for d in os.listdir(data_dir)]
    dirs = [d for d in dirs if path.isdir(d)]
    dirs.sort()
    return dirs


def samples_in_dir(dir_path: str, sample_filter: Callable = None) -> Iterable[DataSample]:
    paths = [path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]
    paths.sort()

    for file_path in paths:
        sample = DataSample(file_path)

        if sample_filter:
            sample = sample_filter(sample)

        if sample:
            yield sample


def samples_for_person(person_name: str, sample_filter: Callable = None) -> Iterable[DataSample]:
    person_dir = path.join(config.Paths.DATASET_DIR, person_name.replace(' ', '_'))
    return samples_in_dir(person_dir, sample_filter=sample_filter)


def all_samples(sample_filter: Callable = None) -> Iterable[DataSample]:
    for dir_path in all_dirs():
        for sample in samples_in_dir(dir_path, sample_filter=sample_filter):
            yield sample


def negative_verification_images(samples: Iterable[DataSample], person_name: str = None,
                                 max_samples: int = sys.maxsize) -> Iterable[np.array]:
    skip_person: str = None
    n_samples = 0

    for sample in samples:

        sample_name = sample.person_name

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
