from .controller import training
from .view.webcam import Webcam

if __name__ == '__main__':
    with Webcam(frame_handler=training.process_frame) as webcam:
        webcam.start_capture()
