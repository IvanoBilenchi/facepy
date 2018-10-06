from face_auth import config


def log(msg: str) -> None:
    if config.DEBUG:
        print(msg)
