from typing import Tuple

from plagiarism_detector import PlagiarismDetector

# Singleton-style detector service to share across blueprints
_detector = PlagiarismDetector()
_model_trained = _detector.is_trained


def get_detector() -> PlagiarismDetector:
    return _detector


def is_model_trained() -> bool:
    return _model_trained


def set_model_trained(value: bool) -> None:
    global _model_trained
    _model_trained = value


def reset_model_service() -> None:
    global _detector, _model_trained
    _detector.reset_model()
    _model_trained = False
