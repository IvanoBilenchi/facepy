import numpy as np

from .video import VideoController
from face_auth import config
from face_auth.model import detector, img
from face_auth.model.geometry import Point, Rect, Size
from face_auth.model.process import Pipeline, Step
from face_auth.view import geometry_renderer
from face_auth.view.video import VideoView


class TrainingController(VideoController):

    # Overrides

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        faces = detector.detect_faces(frame)
        processed = False

        for face in faces:
            landmarks = detector.detect_landmarks(frame, face)

            if not processed and key == VideoView.Key.SPACE:
                processed = True

                rect = Rect.expanded_to_square(landmarks.rect)
                shape = Point.with_new_origin(landmarks.outer_shape, rect.top_left)

                Pipeline.execute('Face extraction', frame, config.DEBUG, [
                    Step('Crop', lambda f: img.cropped(f, rect)),
                    Step('To grayscale', img.to_grayscale),
                    Step('Mask', lambda f: img.masked_to_shape(f, shape)),
                    Step('Resize', lambda f: img.resized(f, Size(150, 150))),
                    Step('Denoise', img.denoised)
                ])

            geometry_renderer.draw_landmarks(frame, landmarks)

        return frame
