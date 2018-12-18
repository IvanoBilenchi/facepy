import numpy as np
from typing import List
from face_auth.model import dataset, preprocess
from face_auth.model.classification import FaceClassifier
from face_auth.model.detector import StaticFaceDetector
from face_auth.model.verification import FaceVerifier


def print_metrics(tp: int, tn: int, fp: int, fn: int) -> None:

    if fp + fn == 0:
        accuracy = 1.0
    else:
        accuracy = (tp + tn) / (tp + fn + fp + tn)

    if fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)

    if fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    if fp == 0:
        specificity = 1.0
    else:
        specificity = tn / (tn + fp)

    if precision == 0.0 and recall == 0.0:
        f1 = float('nan')
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print('Confusion matrix\n----------------')
    print('TP: {}\t\tFP: {}\nFN: {}\t\tTN: {}'.format(tp, fp, fn, tn))
    print('\nOther metrics\n-------------')
    print('Accuracy:   \t{:.2f}'.format(accuracy))
    print('Precision:  \t{:.2f}'.format(precision))
    print('Recall:     \t{:.2f}'.format(recall))
    print('Specificity:\t{:.2f}'.format(specificity))
    print('F1 score:   \t{:.2f}'.format(f1))


def print_metrics_multiclass(cm: np.array, labels: List[str]) -> None:
    n_classes = cm.shape[0]
    n_predictions = np.sum(cm)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.repeat(n_predictions, n_classes) - (tp + fp + fn)

    tp_plus_tn = tp + tn
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    tn_plus_fp = tn + fp
    all_predictions = tp + tn + fp + fn

    accuracy = np.divide(tp_plus_tn, all_predictions,
                         out=np.zeros_like(tp_plus_tn), where=all_predictions != 0)
    precision = np.divide(tp, tp_plus_fp,
                          out=np.zeros_like(tp), where=tp_plus_fp != 0)
    recall = np.divide(tp, tp_plus_fn,
                       out=np.zeros_like(tp), where=tp_plus_fn != 0)
    specificity = np.divide(tn, tn_plus_fp,
                            out=np.zeros_like(tn), where=tn_plus_fp != 0)

    f1 = 2 * precision * recall / (precision + recall)

    print('Confusion matrix\n----------------\n')
    print(cm)
    print_metric('Accuracy', accuracy, labels, tp_plus_fn)
    print_metric('Precision', precision, labels, tp_plus_fn)
    print_metric('Recall', recall, labels, tp_plus_fn)
    print_metric('Specificity', specificity, labels, tp_plus_fn)
    print_metric('F1 score', f1, labels, tp_plus_fn)


def print_metric(name: str, metric: np.array, labels: List[str], predictions: np.array) -> None:
    title = '{} (avg: {:.2f}, w_avg: {:.2f})'.format(name, np.average(metric),
                                                     np.average(metric, weights=predictions))
    print('\n{}\n{}'.format(title, '-' * len(title)))

    for i, m in enumerate(metric):
        print('  - {}: {:.2f}'.format(labels[i], m))


def evaluate_verifier(model_dir: str, skip: int = 0) -> None:
    verifier = FaceVerifier.from_dir(model_dir)
    detector = StaticFaceDetector(scale_factor=1)

    prompt = 'Evaluating verifier for {}'.format(verifier.person_name)
    print(prompt)
    print('-' * len(prompt))

    tn, fn, tp, fp = 0, 0, 0, 0
    skipped = 0

    for sample in dataset.all_samples():
        face_sample = preprocess.data_to_face_sample(detector, sample)

        if not face_sample:
            continue
        elif verifier.person_name == sample.person_name and skipped < skip:
            skipped += 1
            continue

        prediction = verifier.predict(face_sample.image)

        if verifier.person_name == sample.person_name:
            if prediction:
                tp += 1
            else:
                fn += 1
        else:
            if prediction:
                fp += 1
            else:
                tn += 1

        print('{}: {}'.format(sample.file_name, 'verified' if prediction else 'not verified'))

    print('\nEvaluation completed!\n')
    print_metrics(tp, tn, fp, fn)


def evaluate_classifier(model_dir: str, skip: int = 0) -> None:
    classifier = FaceClassifier.from_dir(model_dir)
    detector = StaticFaceDetector(scale_factor=1)

    print('Evaluating classifier for: {}'.format(', '.join(classifier.labels)))
    n_classes = len(classifier.labels)

    cm = np.zeros((n_classes, n_classes), dtype=np.float)

    for i, expected in enumerate(classifier.labels):
        samples = preprocess.data_to_face_samples(detector, dataset.samples_for_person(expected))

        for n_sample, sample in enumerate(samples):
            if n_sample < skip:
                continue

            predicted = classifier.predict(sample.image)

            if predicted:
                j = classifier.labels.index(predicted)
                print('Expected: {} ({}) - Predicted: {} ({})'.format(expected, i,
                                                                      predicted, j))
                cm[i][j] += 1.0

    print('\nEvaluation completed!\n')
    print_metrics_multiclass(cm, classifier.labels)
