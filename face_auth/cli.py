import argparse

from . import config
from .model.input import WebcamStream
from .model.verification import FaceVerifier
from .view.video import VideoView
from .controller.interactive.verification import VerificationVideoController
from .controller.interactive.training import TrainVerifierVideoController


# Subcommands


def train_verifier_sub(args) -> int:
    """train-verifier subcommand."""
    if args.algo:
        config.Recognizer.ALGORITHM = args.algo

    with TrainVerifierVideoController(view=VideoView(), input_stream=WebcamStream()) as controller:
        controller.run_loop()
    return 0


def verify_sub(args) -> int:
    """verify subcommand."""
    del args  # Unused
    with VerificationVideoController(view=VideoView(), input_stream=WebcamStream()) as controller:
        controller.run_loop()
    return 0


# CLI parser


def process_args() -> int:
    """Run actions based on CLI arguments."""
    args = build_parser().parse_args()

    if args.debug:
        config.DEBUG = True

    if args.webcam:
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

    # Config parser
    config_parser = argparse.ArgumentParser(add_help=False)

    group = config_parser.add_argument_group('Configuration')
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
                                   parents=[config_parser, help_parser],
                                   add_help=False)

    group = parser.add_argument_group('Options')
    group.add_argument('-a', '--algo',
                       help='Choose specific algorithm.',
                       choices=[a.name for a in FaceVerifier.Algo],
                       default=config.Recognizer.ALGORITHM)

    parser.set_defaults(func=train_verifier_sub)

    # verify subcommand
    desc = 'Use the trained model to verify the user.'
    parser = subparsers.add_parser('verify',
                                   description=desc,
                                   help=desc,
                                   parents=[config_parser, help_parser],
                                   add_help=False)

    parser.set_defaults(func=verify_sub)

    return main_parser


# Utils


def unsigned_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError('{} is not an unsigned int.'.format(value))
    return ivalue
