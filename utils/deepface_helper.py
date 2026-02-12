"""
DeepFace helper functions for face recognition and analysis.
"""

from typing import Dict, List, Optional, Tuple
from deepface import DeepFace
import os


# Available models in DeepFace
AVAILABLE_MODELS = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

# Default model (ArcFace has best performance according to benchmarks)
DEFAULT_MODEL = "ArcFace"

# Available distance metrics
DISTANCE_METRICS = ["cosine", "euclidean", "euclidean_l2"]


def get_available_models() -> List[str]:
    """
    Get list of available face recognition models.
    
    Returns:
        List of model names
    """
    return AVAILABLE_MODELS


def verify_faces(
    img1_path: str,
    img2_path: str,
    model_name: str = DEFAULT_MODEL,
    distance_metric: str = "cosine",
    detector_backend: str = "opencv"
) -> Dict:
    """
    Verify if two faces belong to the same person.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        model_name: Face recognition model to use
        distance_metric: Distance metric for comparison
        detector_backend: Face detection backend
        
    Returns:
        Dictionary with verification results including:
        - verified: Boolean indicating if same person
        - distance: Calculated distance between faces
        - threshold: Threshold for verification
        - model: Model used
        - similarity_metric: Metric used
    """
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            distance_metric=distance_metric,
            detector_backend=detector_backend,
            enforce_detection=True
        )
        return result
    except Exception as e:
        raise Exception(f"Face verification failed: {str(e)}")


def analyze_face(
    img_path: str,
    actions: Optional[List[str]] = None,
    detector_backend: str = "opencv"
) -> List[Dict]:
    """
    Analyze facial attributes including age, gender, emotion, and race.
    
    Args:
        img_path: Path to image
        actions: List of analysis actions (age, gender, emotion, race)
        detector_backend: Face detection backend
        
    Returns:
        List of dictionaries with analysis results for each detected face
    """
    if actions is None:
        actions = ['age', 'gender', 'race', 'emotion']
    
    try:
        results = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=True
        )
        return results
    except Exception as e:
        raise Exception(f"Face analysis failed: {str(e)}")


def extract_embedding(
    img_path: str,
    model_name: str = DEFAULT_MODEL,
    detector_backend: str = "opencv"
) -> Tuple[List[float], Dict]:
    """
    Extract face embedding vector from an image.
    
    Args:
        img_path: Path to image
        model_name: Face recognition model to use
        detector_backend: Face detection backend
        
    Returns:
        Tuple of (embedding vector, face metadata)
    """
    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True
        )
        
        # DeepFace returns a list of results for each face detected
        if result and len(result) > 0:
            embedding = result[0]["embedding"]
            facial_area = result[0]["facial_area"]
            return embedding, facial_area
        else:
            raise Exception("No faces detected in the image")
            
    except Exception as e:
        raise Exception(f"Embedding extraction failed: {str(e)}")


def detect_faces(
    img_path: str,
    detector_backend: str = "opencv",
    align: bool = True
) -> List[Dict]:
    """
    Detect and extract faces from an image.
    
    Args:
        img_path: Path to image
        detector_backend: Face detection backend
        align: Whether to align faces
        
    Returns:
        List of dictionaries with face data and metadata
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            align=align,
            enforce_detection=True
        )
        return faces
    except Exception as e:
        raise Exception(f"Face detection failed: {str(e)}")


def get_model_info(model_name: str) -> Dict[str, any]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information including embedding dimension
    """
    # Embedding dimensions for each model
    model_dimensions = {
        "VGG-Face": 2622,
        "Facenet": 128,
        "Facenet512": 512,
        "OpenFace": 128,
        "DeepFace": 4096,
        "DeepID": 160,
        "ArcFace": 512,
        "Dlib": 128,
        "SFace": 128,
    }
    
    return {
        "name": model_name,
        "embedding_dimension": model_dimensions.get(model_name, "Unknown")
    }
