import numpy as np

from .video import VideoController
from face_auth import config
from face_auth.model import detector, img
from face_auth.view import geometry_renderer
from face_auth.view.video import VideoView


class TrainingController(VideoController):

    # Overrides

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        faces = detector.detect_faces(frame)

        if faces and key == VideoView.Key.SPACE:
            img.save(img.cropped(frame, faces[-1]), config.Paths.FACE_IMAGE)

        for face in faces:
            geometry_renderer.draw_rect(frame, face)
            landmarks = detector.detect_landmarks(frame, face)
            geometry_renderer.draw_landmarks(frame, landmarks)

        return frame
