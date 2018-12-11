import argparse

from . import config
from .model import fileutils
from .model.recognition_algo import RecognitionAlgo
from .controller.batch import evaluation, training
from .controller.interactive.verification import VerificationVideoController
from .controller.interactive.training import TrainVerifierVideoController


# Subcommands


def train_verifier_sub(args) -> int:
    """train-verifier subcommand."""
    model_dir = config.Paths.VERIFICATION_MODEL_DIR
    samples_dir = args.samples_dir
    algo = RecognitionAlgo[args.algo]

    if samples_dir:
        training.train_verifier(algo, samples_dir, model_dir)
    else:
        with TrainVerifierVideoController(algo, model_dir) as controller:
            controller.run_loop()

    return 0


def train_classifier_sub(args) -> int:
    """train-classifier subcommand."""
    model_dir = config.Paths.CLASSIFICATION_MODEL_DIR
    algo = RecognitionAlgo[args.algo]

    training.train_classifier(algo, model_dir,
                              min_samples=args.min_samples,
                              training_samples=args.training_samples)
    return 0


def verify_sub(args) -> int:
    """verify subcommand."""
    del args  # Unused
    with VerificationVideoController(config.Paths.VERIFICATION_MODEL_DIR) as controller:
        controller.run_loop()
    return 0


def evaluate_verifier_sub(args) -> int:
    """evaluate-verifier subcommand."""
    del args  # Unused
    evaluation.evaluate_verifier(config.Paths.VERIFICATION_MODEL_DIR)
    return 0


def evaluate_classifier_sub(args) -> int:
    """evaluate-classifier subcommand."""
    evaluation.evaluate_classifier(config.Paths.CLASSIFICATION_MODEL_DIR, skip=args.skip)
    return 0


# CLI parser


def process_args() -> int:
    """Run actions based on CLI arguments."""
    args = build_parser().parse_args()

    if args.debug:
        config.DEBUG = True

    if config.DEBUG:
        fileutils.delete_dir(config.Paths.DEBUG_DIR)
        fileutils.create_dir(config.Paths.DEBUG_DIR)

    if hasattr(args, 'webcam') and args.webcam is not None:
        config.WEBCAM = args.webcam

    if not hasattr(args, 'func'):
        raise ValueError(('Invalid argument(s). Please run "face_auth -h" or '
                          '"face_auth <subcommand> -h" for help.'))

    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI parser."""

    # Help parser
    help_parser = argparse.ArgumentParser(add_help=False)

    group = help_parser.add_argument_group('Help and debug')
    group.add_argument('--debug',
                       help='Enable debug output.',
                       action='store_true')
    group.add_argument('-h', '--help',
                       help='Show this help message and exit.',
                       action='help')

    # Webcam parser
    webcam_parser = argparse.ArgumentParser(add_help=False)

    group = webcam_parser.add_argument_group('Webcam configuration')
    group.add_argument('-w', '--webcam',
                       help='Select a specific webcam.',
                       type=unsigned_int,
                       default=config.WEBCAM)

    # Main parser
    main_parser = argparse.ArgumentParser(prog='face_auth',
                                          description='Facial recognition framework.',
                                          parents=[help_parser],
                                          add_help=False)

    subparsers = main_parser.add_subparsers(title='Available commands')

    # train-verifier subcommand
    desc = 'Train a face verifier.'
    parser = subparsers.add_parser('train-verifier',
                                   description=desc,
                                   help=desc,
                                   parents=[webcam_parser, help_parser],
                                   add_help=False)

    group = parser.add_argument_group('Options')
    group.add_argument('-a', '--algo',
                       help='Use a specific algorithm.',
                       choices=[a.name for a in RecognitionAlgo],
                       default=config.Recognizer.ALGORITHM)
    group.add_argument('-d', '--samples_dir',
                       help='Trains a verifier with images from the specified dir.')

    parser.set_defaults(func=train_verifier_sub)

    # train-classifier subcommand
    desc = 'Train a face classifier.'
    parser = subparsers.add_parser('train-classifier',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser],
                                   add_help=False)

    group = parser.add_argument_group('Options')
    group.add_argument('-a', '--algo',
                       help='Use a specific algorithm.',
                       choices=[a.name for a in RecognitionAlgo],
                       default=config.Recognizer.ALGORITHM)
    group.add_argument('-m', '--min-samples',
                       help='Train classifier for people having at least as many samples.',
                       type=unsigned_int,
                       default=config.Recognizer.CLASSIFICATION_MIN_SAMPLES)
    group.add_argument('-t', '--training-samples',
                       help='Number of training samples per person.',
                       type=unsigned_int,
                       default=config.Recognizer.CLASSIFICATION_TRAINING_SAMPLES)

    parser.set_defaults(func=train_classifier_sub)

    # verify subcommand
    desc = 'Use the trained model to verify the user.'
    parser = subparsers.add_parser('verify',
                                   description=desc,
                                   help=desc,
                                   parents=[webcam_parser, help_parser],
                                   add_help=False)

    parser.set_defaults(func=verify_sub)

    # evaluate-verifier subcommand
    desc = 'Evaluate a trained verifier.'
    parser = subparsers.add_parser('evaluate-verifier',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser],
                                   add_help=False)

    parser.set_defaults(func=evaluate_verifier_sub)

    # evaluate-classifier subcommand
    desc = 'Evaluate a trained classifier.'
    parser = subparsers.add_parser('evaluate-classifier',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser],
                                   add_help=False)

    group = parser.add_argument_group('Options')
    group.add_argument('-s', '--skip',
                       help='Skip first samples for each class (account for training data).',
                       type=unsigned_int,
                       default=0)

    parser.set_defaults(func=evaluate_classifier_sub)

    return main_parser


# Utils


def unsigned_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError('{} is not an unsigned int.'.format(value))
    return ivalue
