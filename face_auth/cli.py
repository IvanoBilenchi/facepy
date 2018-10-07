import argparse

from . import config
from .controller.training import TrainingController
from .model.input import WebcamStream
from .view.video import VideoView


# Subcommands


def training_sub(args) -> int:
    """Training subcommand."""
    del args  # Unused
    with TrainingController(view=VideoView(), input_stream=WebcamStream()) as controller:
        controller.run_loop()
    return 0


# CLI parser


def process_args() -> int:
    """Run actions based on CLI arguments."""
    args = build_parser().parse_args()

    if args.debug:
        config.DEBUG = True

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

    # Main parser
    main_parser = argparse.ArgumentParser(prog='face_auth',
                                          description='Facial recognition framework.',
                                          parents=[help_parser],
                                          add_help=False)

    subparsers = main_parser.add_subparsers(title='Available commands')

    # Training subcommand
    desc = 'Train the model to recognize a face.'
    parser_classification = subparsers.add_parser('training',
                                                  description=desc,
                                                  help=desc,
                                                  parents=[help_parser],
                                                  add_help=False)

    parser_classification.set_defaults(func=training_sub)

    return main_parser