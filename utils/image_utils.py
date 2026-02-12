"""
Image utility functions for Streamlit application.
"""

import os
import tempfile
import streamlit as st
from PIL import Image
from typing import Optional
import shutil


# Create temp directory for uploaded files
TEMP_DIR = os.path.join(tempfile.gettempdir(), "face_recognition_temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit uploaded file to temporary storage.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to saved file
    """
    try:
        # Create a unique filename
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        raise Exception(f"Failed to save uploaded file: {str(e)}")


def display_image_with_info(image_path: str, caption: str = "", width: Optional[int] = None):
    """
    Display an image in Streamlit with optional caption.
    
    Args:
        image_path: Path to image file
        caption: Caption to display below image
        width: Optional width for the image
    """
    try:
        image = Image.open(image_path)
        if width:
            st.image(image, caption=caption, width=width)
        else:
            st.image(image, caption=caption, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to display image: {str(e)}")


def create_comparison_view(img1_path: str, img2_path: str, result: dict):
    """
    Create a side-by-side comparison view for face verification.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        result: Verification result dictionary
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image 1")
        display_image_with_info(img1_path)
    
    with col2:
        st.subheader("Image 2")
        display_image_with_info(img2_path)
    
    # Display results
    st.markdown("---")
    if result.get("verified", False):
        st.success("✅ **Verification Result: SAME PERSON**")
    else:
        st.error("❌ **Verification Result: DIFFERENT PEOPLE**")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Distance", f"{result.get('distance', 0):.4f}")
    with col2:
        st.metric("Threshold", f"{result.get('threshold', 0):.4f}")
    with col3:
        confidence = (1 - result.get('distance', 0) / result.get('threshold', 1)) * 100
        confidence = max(0, min(100, confidence))
        st.metric("Confidence", f"{confidence:.1f}%")


def cleanup_temp_files():
    """
    Clean up temporary files in the temp directory.
    """
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to clean up temp files: {str(e)}")


def format_emotion_results(emotion_dict: dict) -> str:
    """
    Format emotion analysis results for display.
    
    Args:
        emotion_dict: Dictionary of emotions and their scores
        
    Returns:
        Formatted string with emotions sorted by score
    """
    sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
    result = []
    for emotion, score in sorted_emotions:
        result.append(f"**{emotion.capitalize()}**: {score:.2f}%")
    return "\n".join(result)


def get_image_size(image_path: str) -> tuple:
    """
    Get the dimensions of an image.
    
    Args:
        image_path: Path to image
        
    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        return (0, 0)
