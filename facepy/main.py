import sys
from . import cli, config


# Main


def main() -> int:
    try:
        ret_val = cli.process_args()
    except KeyboardInterrupt:
        print('Interrupted by user.')
        ret_val = 1
    except Exception as e:
        if config.DEBUG:
            raise
        else:
            print(str(e))
            ret_val = 1

    return ret_val


if __name__ == '__main__':
    sys.exit(main())
