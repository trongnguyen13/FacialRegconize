"""
Face Recognition & Analysis Web Application
Built with Streamlit and DeepFace
"""

import streamlit as st
import os
from datetime import datetime
import uuid

# Import utility functions
from utils.deepface_helper import (
    verify_faces,
    analyze_face,
    extract_embeddings,
    detect_faces,
    get_available_models,
    get_model_info
)
from utils.image_utils import (
    save_uploaded_file,
    display_image_with_info,
    create_comparison_view,
    cleanup_temp_files,
    format_emotion_results
)
from utils.pinecone_helper import initialize_pinecone_from_env

# Page configuration
st.set_page_config(
    page_title="Face Recognition App",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'pinecone_helper' not in st.session_state:
        st.session_state.pinecone_helper = None
    if 'pinecone_initialized' not in st.session_state:
        st.session_state.pinecone_initialized = False


def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">👤 Face Recognition & Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by DeepFace & Pinecone")
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png", 
                 width=200)
        st.markdown("## Navigation")
        
        # Feature selection
        feature = st.radio(
            "Select Feature",
            ["🏠 Home", "✅ Face Verification", "🔍 Facial Analysis", 
             "🔎 Face Search", "➕ Register Face", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model selection (for features that use it)
        st.markdown("### Settings")
        models = get_available_models()
        selected_model = st.selectbox("Face Recognition Model", models, index=models.index("ArcFace"))
        
        # Display model info
        model_info = get_model_info(selected_model)
        st.info(f"**Embedding Dimension:** {model_info['embedding_dimension']}")
        
        st.markdown("---")
        
        # Pinecone status
        st.markdown("### Pinecone Status")
        if st.session_state.pinecone_helper:
            st.success("✅ Connected")
            try:
                stats = st.session_state.pinecone_helper.get_stats()
                st.metric("Stored Faces", stats['total_vectors'])
            except:
                st.warning("Unable to fetch stats")
        else:
            st.error("❌ Not Connected")
            st.caption("Add API key to .env file")
        
        return feature, selected_model


def render_home():
    """Render the home page."""
    st.markdown("## Welcome to Face Recognition & Analysis App! 👋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Features
        
        - **Face Verification**: Compare two faces to verify identity
        - **Facial Analysis**: Detect age, gender, emotion, and race
        - **Face Search**: Search for similar faces in your database
        - **Face Registration**: Add new faces to your database
        
        ### 🚀 Getting Started
        
        1. Select a feature from the sidebar
        2. Upload your image(s)
        3. Click the action button
        4. View the results!
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Supported Models
        
        - VGG-Face
        - **ArcFace** (Recommended)
        - Facenet / Facenet512
        - OpenFace
        - DeepFace
        - DeepID
        - Dlib
        - SFace
        
        ### 🔧 Configuration
        
        To use Face Search and Registration features, you need to configure Pinecone:
        
        1. Create a `.env` file based on `.env.example`
        2. Add your Pinecone API key
        3. Restart the application
        """)
    
    st.markdown("---")
    st.info("👈 **Select a feature from the sidebar to get started!**")


def render_face_verification(model_name):
    """Render the face verification feature."""
    st.markdown("## ✅ Face Verification")
    st.markdown("Compare two images to verify if they belong to the same person.")
    
    st.markdown("### 📷 Choose Input Method")
    input_method = st.radio(
        "Select how to capture images:",
        ["📁 Upload Image Files", "📸 Use Camera"],
        horizontal=True,
        key="verify_input_method",
        label_visibility="collapsed"
    )
    
    img1_path = None
    img2_path = None
    
    col1, col2 = st.columns(2)
    
    # File upload method
    if input_method == "📁 Upload Image Files":
        with col1:
            st.markdown("**First Image**")
            uploaded_file1 = st.file_uploader("Upload First Image", type=['jpg', 'jpeg', 'png'], key="verify_img1", label_visibility="collapsed")
            if uploaded_file1:
                img1_path = save_uploaded_file(uploaded_file1)
                display_image_with_info(img1_path, width=300)
        
        with col2:
            st.markdown("**Second Image**")
            uploaded_file2 = st.file_uploader("Upload Second Image", type=['jpg', 'jpeg', 'png'], key="verify_img2", label_visibility="collapsed")
            if uploaded_file2:
                img2_path = save_uploaded_file(uploaded_file2)
                display_image_with_info(img2_path, width=300)
    
    # Camera capture method
    else:
        with col1:
            st.markdown("**📸 First Person**")
            camera_photo1 = st.camera_input("Capture first image", key="camera_verify1", label_visibility="collapsed")
            if camera_photo1:
                img1_path = save_uploaded_file(camera_photo1)
        
        with col2:
            st.markdown("**📸 Second Person**")
            camera_photo2 = st.camera_input("Capture second image", key="camera_verify2", label_visibility="collapsed")
            if camera_photo2:
                img2_path = save_uploaded_file(camera_photo2)
    
    # Verify button and results
    if img1_path and img2_path:
        if st.button("🔍 Verify Faces", type="primary"):
            with st.spinner("Analyzing faces..."):
                try:
                    # Verify faces
                    result = verify_faces(img1_path, img2_path, model_name=model_name)
                    
                    # Display results
                    st.markdown("---")
                    create_comparison_view(img1_path, img2_path, result)
                    
                    # Additional details
                    with st.expander("📋 Detailed Results"):
                        st.json(result)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        if input_method == "📁 Upload Image Files":
            st.info("📸 Please upload both images to proceed.")
        else:
            st.info("📸 Please capture both photos using the camera to proceed.")


def render_facial_analysis(model_name):
    """Render the facial analysis feature."""
    st.markdown("## 🔍 Facial Attribute Analysis")
    st.markdown("Analyze age, gender, emotion, and race from a face image.")
    
    st.markdown("### 📷 Choose Input Method")
    input_method = st.radio(
        "Select how to capture the image:",
        ["📁 Upload Image File", "📸 Use Camera"],
        horizontal=True,
        key="analysis_input_method",
        label_visibility="collapsed"
    )
    
    img_path = None
    
    # File upload method
    if input_method == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="analysis_img", label_visibility="collapsed")
        if uploaded_file:
            img_path = save_uploaded_file(uploaded_file)
    
    # Camera capture method
    else:
        st.markdown("📸 **Camera Capture**")
        camera_photo = st.camera_input("Capture image", key="camera_analysis", label_visibility="collapsed")
        if camera_photo:
            img_path = save_uploaded_file(camera_photo)
    
    if img_path:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Input Image")
            display_image_with_info(img_path)
        
        if st.button("🔍 Analyze Face", type="primary"):
            with st.spinner("Analyzing facial attributes..."):
                try:
                    results = analyze_face(img_path)
                    
                    with col2:
                        st.markdown("### 📊 Analysis Results")
                        
                        for idx, face_result in enumerate(results):
                            if len(results) > 1:
                                st.markdown(f"#### Face {idx + 1}")
                            
                            # Display results in metrics
                            metric_col1, metric_col2 = st.columns(2)
                            
                            with metric_col1:
                                st.metric("👤 Age", f"{face_result.get('age', 'N/A')} years")
                                st.metric("⚧ Gender", face_result.get('dominant_gender', 'N/A').capitalize())
                            
                            with metric_col2:
                                st.metric("😊 Emotion", face_result.get('dominant_emotion', 'N/A').capitalize())
                                st.metric("🌍 Race", face_result.get('dominant_race', 'N/A').capitalize())
                            
                            # Emotion breakdown
                            st.markdown("#### 😊 Emotion Breakdown")
                            emotion_data = face_result.get('emotion', {})
                            st.markdown(format_emotion_results(emotion_data))
                            
                            # Detailed results
                            with st.expander("📋 Full Analysis Data"):
                                st.json(face_result)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        if input_method == "📁 Upload Image File":
            st.info("📸 Please upload an image to proceed.")
        else:
            st.info("📸 Please capture a photo using the camera to proceed.")


def render_face_search(model_name):
    """Render the face search feature."""
    st.markdown("## 🔎 Face Search")
    st.markdown("Search for similar faces in your Pinecone database.")

    if not st.session_state.pinecone_helper:
        st.warning("⚠️ Pinecone is not configured. Please add your API key to the `.env` file and restart the app.")
        st.code("""
# .env file
PINECONE_API_KEY=your_api_key_here
PINECONE_INDEX_NAME=face-recognition-index
        """)
        return

    st.markdown("### 📷 Choose Input Method")
    input_method = st.radio(
        "Select how to capture the query image:",
        ["📁 Upload Image File", "📸 Use Camera"],
        horizontal=True,
        key="search_input_method",
        label_visibility="collapsed"
    )

    img_path = None

    # File upload method
    if input_method == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload Query Image", type=['jpg', 'jpeg', 'png'], key="search_img", label_visibility="collapsed")
        if uploaded_file:
            img_path = save_uploaded_file(uploaded_file)

    # Camera capture method
    else:
        st.markdown("📸 **Camera Capture**")
        camera_photo = st.camera_input("Capture query image", key="camera_search", label_visibility="collapsed")
        if camera_photo:
            img_path = save_uploaded_file(camera_photo)

    col1, col2 = st.columns([1, 2])

    with col1:
        top_k = st.slider("Number of Results", 1, 10, 5)
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05)

    if img_path:
        st.markdown("### 📸 Query Image")
        display_image_with_info(img_path, width=300)

        try:
            face_candidates = extract_embeddings(img_path, model_name=model_name)
        except Exception as e:
            st.error(f"❌ Error detecting faces: {str(e)}")
            return

        selected_face_idx = 0
        if len(face_candidates) > 1:
            st.info(f"Detected **{len(face_candidates)}** faces. Please select one for search.")
            selected_face_idx = st.selectbox(
                "Choose a face",
                options=list(range(len(face_candidates))),
                format_func=lambda idx: f"Face {idx + 1}",
                key="search_face_selector"
            )

        selected_face = face_candidates[selected_face_idx]

        if st.button("🔍 Search Similar Faces", type="primary"):
            with st.spinner("Searching..."):
                try:
                    # Search in Pinecone with selected face embedding
                    matches = st.session_state.pinecone_helper.search_faces(
                        query_embedding=selected_face["embedding"],
                        top_k=top_k,
                        score_threshold=threshold
                    )

                    st.markdown("---")
                    st.markdown(f"### 🎯 Search Results (Face {selected_face_idx + 1})")

                    if matches:
                        for idx, match in enumerate(matches):
                            with st.expander(f"Match {idx + 1} - Similarity: {match['score']:.4f}"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Face ID", match['id'])
                                    st.metric("Similarity Score", f"{match['score']:.4f}")
                                with col_b:
                                    st.json(match['metadata'])
                    else:
                        st.info("No matches found above the similarity threshold.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        if input_method == "📁 Upload Image File":
            st.info("📸 Please upload an image to search.")
        else:
            st.info("📸 Please capture a photo using the camera to search.")


def render_face_registration(model_name):
    """Render the face registration feature."""
    st.markdown("## ➕ Register Face")
    st.markdown("Add a new face to your Pinecone database.")

    if not st.session_state.pinecone_helper:
        st.warning("⚠️ Pinecone is not configured. Please add your API key to the `.env` file and restart the app.")
        return

    # Choose input method
    st.markdown("### 📷 Choose Input Method")
    input_method = st.radio(
        "Select how to capture the face:",
        ["📁 Upload Image File", "📸 Use Camera"],
        horizontal=True,
        label_visibility="collapsed"
    )

    img_path = None

    # File upload method
    if input_method == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload Face Image", type=['jpg', 'jpeg', 'png'], key="register_img")
        if uploaded_file:
            img_path = save_uploaded_file(uploaded_file)

    # Camera capture method
    else:
        st.markdown("📸 **Camera Capture**")
        st.caption("Click the camera button below to capture a photo")

        camera_photo = st.camera_input("Take a picture", key="camera_register")

        if camera_photo:
            img_path = save_uploaded_file(camera_photo)

    # Process the captured/uploaded image
    if img_path:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📸 Image to Register")
            display_image_with_info(img_path)

        try:
            face_candidates = extract_embeddings(img_path, model_name=model_name)
        except Exception as e:
            st.error(f"❌ Error detecting faces: {str(e)}")
            return

        selected_face_idx = 0
        if len(face_candidates) > 1:
            st.info(f"Detected **{len(face_candidates)}** faces. Please select one to register.")
            selected_face_idx = st.selectbox(
                "Choose a face",
                options=list(range(len(face_candidates))),
                format_func=lambda idx: f"Face {idx + 1}",
                key="register_face_selector"
            )

        selected_face = face_candidates[selected_face_idx]

        with col2:
            st.markdown("### 📝 Face Information")

            # Metadata inputs
            face_id = st.text_input("Face ID (leave empty for auto-generation)", "")
            person_name = st.text_input("Person Name", "")
            notes = st.text_area("Additional Notes", "")

            if st.button("➕ Register Face", type="primary"):
                with st.spinner("Registering face..."):
                    try:
                        # Generate ID if not provided
                        if not face_id:
                            face_id = f"face_{uuid.uuid4().hex[:8]}"

                        # Prepare metadata
                        metadata = {
                            "name": person_name,
                            "notes": notes,
                            "registered_at": datetime.now().isoformat(),
                            "model": model_name,
                            "capture_method": "camera" if input_method == "📸 Use Camera" else "upload",
                            "source_face_index": selected_face_idx,
                            "source_faces_detected_count": len(face_candidates)
                        }

                        # Register in Pinecone
                        success = st.session_state.pinecone_helper.register_face(
                            embedding=selected_face["embedding"],
                            face_id=face_id,
                            metadata=metadata
                        )

                        if success:
                            st.success(f"✅ Face registered successfully with ID: `{face_id}`")
                            st.balloons()

                            with st.expander("📋 Registration Details"):
                                st.json({
                                    "face_id": face_id,
                                    "embedding_dimension": len(selected_face["embedding"]),
                                    "facial_area": selected_face["facial_area"],
                                    "metadata": metadata
                                })

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    else:
        if input_method == "📁 Upload Image File":
            st.info("📸 Please upload an image to register.")
        else:
            st.info("📸 Please capture a photo using the camera to register.")


def render_about():
    """Render the about page."""
    st.markdown("## ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    
    This application demonstrates the capabilities of modern face recognition and analysis using:
    
    - **DeepFace**: A lightweight face recognition and facial attribute analysis library
    - **Pinecone**: A vector database for efficient similarity search
    - **Streamlit**: An open-source framework for building data applications
    
    ### 🔬 Technology Stack
    
    - **Python 3.13+**
    - **DeepFace Library**: Face recognition and analysis
    - **Pinecone**: Vector database for face embeddings
    - **Streamlit**: Web application framework
    - **OpenCV**: Image processing
    - **TensorFlow/Keras**: Deep learning backend
    
    ### 📚 Resources
    
    - [DeepFace GitHub](https://github.com/serengil/deepface)
    - [Pinecone Documentation](https://docs.pinecone.io/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    
    ### ⚖️ Privacy & Ethics
    
    This application is for educational and demonstration purposes. When using face recognition technology:
    
    - Always obtain consent before capturing/analyzing faces
    - Ensure compliance with privacy laws (GDPR, CCPA, etc.)
    - Handle biometric data with appropriate security measures
    - Be aware of potential biases in facial recognition models
    
    ### 👨‍💻 Development
    
    Built with ❤️ using modern AI and web technologies.
    """)


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Try to initialize Pinecone
    if not st.session_state.pinecone_initialized:
        try:
            st.session_state.pinecone_helper = initialize_pinecone_from_env()
            st.session_state.pinecone_initialized = True
        except Exception as e:
            st.session_state.pinecone_helper = None
            st.session_state.pinecone_initialized = True
    
    # Render header
    render_header()
    
    # Render sidebar and get selected feature
    feature, model_name = render_sidebar()
    
    # Render selected feature
    if feature == "🏠 Home":
        render_home()
    elif feature == "✅ Face Verification":
        render_face_verification(model_name)
    elif feature == "🔍 Facial Analysis":
        render_facial_analysis(model_name)
    elif feature == "🔎 Face Search":
        render_face_search(model_name)
    elif feature == "➕ Register Face":
        render_face_registration(model_name)
    elif feature == "ℹ️ About":
        render_about()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Face Recognition App | Powered by DeepFace & Pinecone | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
