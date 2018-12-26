from typing import Iterable, Optional

import numpy as np

from . import img
from .dataset import DataSample
from .detector import FaceDetector, FaceSample
from .geometry import Size
from .process import Pipeline, Step


def prepare_for_recognition(sample: FaceSample,
                            preprocess: bool = True, debug: bool = False) -> np.array:
    """
    Prepares the given face sample for recognition.

    If 'preprocess' is true, then the image is preprocessed according to this pipeline:
        - Conversion to grayscale.
        - Masking in order to remove the background.
        - Alignment based on the position of the eyes.
        - Resizing, denoising, equalization and normalization.
    Otherwise, the image is just resized.

    Some algorithms (such as the geometric regognizer) either do not require preprocessing,
    or they do it internally (Dlib's CNN recognizer falls into this category).
    """

    if preprocess:
        rect = sample.face.landmarks.square()
        shape = sample.face.landmarks.thin_shape
        matrix = sample.face.landmarks.alignment_matrix()

        steps = [
            Step('To grayscale', img.to_grayscale),
            Step('Mask', lambda f: img.masked_to_shape(f, shape)),
            Step('Align', lambda f: img.transform(f, matrix, rect.size)),
            Step('Resize', lambda f: img.resized(f, Size(100, 100))),
            Step('Denoise', img.denoised),
            Step('Equalize', img.equalized),
            Step('Normalize', img.normalized)
        ]
    else:
        steps = [Step('Resize', lambda f: img.resized(f, Size(250, 250)))]

    return Pipeline.execute('Preprocess face', sample.image, debug, steps)


def data_to_face_sample(detector: FaceDetector, sample: DataSample) -> Optional[FaceSample]:
    """Converts the given data sample into a face sample."""
    face_sample = detector.get_main_face_sample(sample.image)
    return face_sample if face_sample and face_sample.pose_is_frontal() else None


def data_to_face_samples(detector: FaceDetector,
                         samples: Iterable[DataSample]) -> Iterable[FaceSample]:
    """Converts the given data samples into face samples."""
    for sample in samples:
        face_sample = data_to_face_sample(detector, sample)
        if face_sample:
            yield face_sample
