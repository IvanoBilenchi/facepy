from .controller import training
from .view import webcam

if __name__ == '__main__':
    webcam.start_capture(training.process_frame)
