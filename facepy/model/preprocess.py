import numpy as np
from typing import Iterable, Optional

from . import img
from .dataset import DataSample
from .detector import FaceDetector, FaceSample
from .geometry import Size
from .process import Pipeline, Step


def extract_frontal_face(detector: FaceDetector, image: np.array,
                         preprocess: bool = True, debug: bool = False) -> Optional[np.array]:
    sample = detector.extract_main_face_sample(image)

    if sample is None or not sample.pose_is_frontal():
        return None

    return extract_face(sample, preprocess=preprocess, debug=debug)


def extract_face(sample: FaceSample, preprocess: bool = True, debug: bool = False) -> np.array:

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

    return Pipeline.execute('Face extraction', sample.image, debug, steps)


def data_to_face_sample(detector: FaceDetector, sample: DataSample) -> Optional[FaceSample]:
    face_sample = detector.extract_main_face_sample(sample.image)
    return face_sample if face_sample and face_sample.pose_is_frontal() else None


def data_to_face_samples(detector: FaceDetector,
                         samples: Iterable[DataSample]) -> Iterable[FaceSample]:
    for sample in samples:
        face_sample = data_to_face_sample(detector, sample)
        if face_sample:
            yield face_sample
