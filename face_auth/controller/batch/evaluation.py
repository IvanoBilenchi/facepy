from face_auth.model import dataset
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
        f1 = round(2 * precision * recall / (precision + recall))

    print('\nEvaluation completed!\n')
    print('Confusion matrix\n----------------')
    print('TP: {}\t\tFP: {}\nFN: {}\t\tTN: {}'.format(tp, fp, fn, tn))
    print('\nOther metrics\n-------------')
    print('Accuracy:   \t{:.2f}'.format(accuracy))
    print('Precision:  \t{:.2f}'.format(precision))
    print('Recall:     \t{:.2f}'.format(recall))
    print('Specificity:\t{:.2f}'.format(specificity))
    print('F1 score:   \t{:.2f}'.format(f1))


def evaluate_verifier(model_dir: str) -> None:
    verifier = FaceVerifier.from_dir(model_dir)

    prompt = 'Evaluating verifier for {}'.format(verifier.person_name)
    print(prompt)
    print('-' * len(prompt))

    tn, fn, tp, fp = 0, 0, 0, 0

    for sample in dataset.all_samples():
        result = verifier.predict(sample.image)

        if verifier.person_name == sample.person_name:
            if result:
                tp += 1
            else:
                fn += 1
        else:
            if result:
                fp += 1
            else:
                tn += 1

        print('{}: {}'.format(sample.file_name, 'verified' if result else 'not verified'))

    print_metrics(tp, tn, fp, fn)
