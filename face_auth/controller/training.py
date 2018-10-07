import numpy as np

from .video import VideoController
from face_auth import config
from face_auth.model import detector, img
from face_auth.model.geometry import Size
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

                rect = landmarks.square()
                shape = landmarks.outer_shape
                matrix = landmarks.alignment_matrix()

                Pipeline.execute('Face extraction', frame, config.DEBUG, [
                    Step('To grayscale', img.to_grayscale),
                    Step('Mask', lambda f: img.masked_to_shape(f, shape)),
                    Step('Align', lambda f: img.transform(f, matrix, rect.size)),
                    Step('Resize', lambda f: img.resized(f, Size(150, 150))),
                    Step('Denoise', img.denoised),
                    Step('Normalize', img.normalized),
                ])

            geometry_renderer.draw_landmarks(frame, landmarks)

        return frame
