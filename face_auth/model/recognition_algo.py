from enum import Enum


class RecognitionAlgo(Enum):
    EIGEN = 0
    FISHER = 1
    LBPH = 2
    EUCLIDEAN = 3
    CNN = 4
