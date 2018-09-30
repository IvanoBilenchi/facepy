from .controller.training import TrainingController
from .model.input import WebcamStream
from .view.video import VideoView

if __name__ == '__main__':
    with TrainingController(view=VideoView(), input_stream=WebcamStream()) as controller:
        controller.run_loop()
