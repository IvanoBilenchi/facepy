from face_auth.model import dataset
from face_auth.model.verification import FaceVerifier


def train_verifier(algo: FaceVerifier.Algo, samples_dir: str, model_dir: str) -> None:
    verifier = FaceVerifier.create(algo)
    verifier.person_name = dataset.person_name_from_dir(samples_dir)
    samples = list(dataset.samples_in_dir(samples_dir))
    verifier.train(samples)
    verifier.save(model_dir)
