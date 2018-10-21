import csv
import cv2.cv2 as cv2
import numpy as np
from os import path
from typing import Callable, Iterable, Optional


class Dataset:

    def __init__(self, data_dir: str, training_set_tsv: str) -> None:
        self.data_dir = data_dir
        self.training_set_tsv = training_set_tsv

    def file_path(self, person: str, index: int = 1) -> str:
        file_name = '{}_{:04d}.jpg'.format(person, index)
        return path.join(self.data_dir, person, file_name)

    def get_image(self, person: str, index: int = 1) -> np.array:
        return cv2.imread(self.file_path(person, index))

    def training_samples(self, preprocessor: Callable[[np.array],
                                                      Optional[np.array]]) -> Iterable[np.array]:
        with open(self.training_set_tsv, 'r') as tsv:
            for row in csv.reader(tsv, delimiter='\t'):
                try:
                    image = preprocessor(self.get_image(row[0], int(row[1])))
                    if image is not None:
                        yield image
                except IndexError:
                    continue
