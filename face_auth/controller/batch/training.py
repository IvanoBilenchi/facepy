from typing import Iterable

from face_auth.model import dataset
from face_auth.model.classification import FaceClassifier
from face_auth.model.dataset import DataSample
from face_auth.model.detector import FaceSample, StaticFaceDetector
from face_auth.model.recognition_algo import RecognitionAlgo
from face_auth.model.verification import FaceVerifier


def train_verifier(algo: RecognitionAlgo, samples_dir: str, model_dir: str) -> FaceVerifier:
    detector = StaticFaceDetector(scale_factor=1)

    verifier = FaceVerifier.create(algo)
    verifier.person_name = dataset.person_name_from_dir(samples_dir)
    samples = list(_data_to_face_samples(detector, dataset.samples_in_dir(samples_dir)))

    verifier.train(samples)
    verifier.save(model_dir)

    print('Successfully trained verifier!')

    return verifier


def train_classifier(algo: RecognitionAlgo, model_dir: str,
                     min_samples: int = 20, training_samples: int = 10) -> FaceClassifier:
    detector = StaticFaceDetector(scale_factor=1)
    classifier = FaceClassifier.create(algo)
    data = {}

    for dir_path in dataset.all_dirs():
        data_samples = list(dataset.samples_in_dir(dir_path))

        if len(data_samples) < min_samples:
            continue

        samples = list(_data_to_face_samples(detector, data_samples))

        if len(samples) < min_samples:
            continue

        person_name = dataset.person_name_from_dir(dir_path)
        data[person_name] = samples[:training_samples]
        print('Added training data for {}'.format(person_name))

    classifier.train(data)
    classifier.save(model_dir)

    print('Successfully trained classifier!')

    return classifier


# Utils


def _data_to_face_samples(detector: StaticFaceDetector,
                          samples: Iterable[DataSample]) -> Iterable[FaceSample]:
    for sample in samples:
        face_sample = detector.extract_main_face_sample(sample.image)
        if face_sample:
            yield face_sample
