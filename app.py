"""
Staff Directory Web Application
Built with Streamlit and DeepFace
"""

import streamlit as st
from datetime import datetime
import uuid

# Import utility functions
from utils.deepface_helper import extract_embedding
from utils.image_utils import (
    save_uploaded_file,
    display_image_with_info,
)
from utils.pinecone_helper import initialize_pinecone_from_env

# Fixed model — ArcFace (best accuracy, 512-dim embeddings)
MODEL_NAME = "ArcFace"

# Page configuration
st.set_page_config(
    page_title="Staff Directory",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    .staff-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .staff-card h3 {
        margin: 0 0 0.5rem 0;
        word-break: break-word;
    }
    .staff-card p {
        margin: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
            padding: 0.5rem 0;
        }
        .staff-card {
            padding: 1rem;
        }
        /* Make Streamlit buttons easier to tap */
        .stButton > button {
            min-height: 44px;
            width: 100%;
        }
        /* Full-width file uploader on mobile */
        .stFileUploader {
            width: 100%;
        }
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
    st.markdown('<h1 class="main-header">👥 Staff Directory</h1>', unsafe_allow_html=True)
    st.markdown("### Identify your colleagues instantly")
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with navigation."""
    with st.sidebar:
        st.markdown("## Navigation")

        feature = st.radio(
            "Select Feature",
            ["🏠 Home", "🔎 Find Staff", "➕ Register Staff", "📋 Staff Directory"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Pinecone status
        st.markdown("### Database Status")
        if st.session_state.pinecone_helper:
            st.success("Connected")
            try:
                stats = st.session_state.pinecone_helper.get_stats()
                st.metric("Registered Staff", stats['total_vectors'])
            except:
                st.warning("Unable to fetch stats")
        else:
            st.error("Not Connected")
            st.caption("Add API key to .env file")

        return feature


def render_home():
    """Render the home page."""
    st.markdown("## Welcome to Staff Directory! 👋")

    st.markdown("""
    This app helps you get to know your colleagues. Take a photo of someone you meet
    and instantly find out their name, role, and department.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🔎 Find Staff
        Take a photo or upload an image to identify a colleague.
        """)

    with col2:
        st.markdown("""
        #### ➕ Register Staff
        Register your face with your name, role, and department.
        """)

    with col3:
        st.markdown("""
        #### 📋 Staff Directory
        Browse all registered staff members.
        """)

    st.markdown("---")

    if st.session_state.pinecone_helper:
        try:
            stats = st.session_state.pinecone_helper.get_stats()
            st.metric("Total Registered Staff", stats['total_vectors'])
        except:
            pass
    else:
        st.warning("Database is not connected. Please configure your `.env` file to enable all features.")

    st.info("👈 **Select a feature from the sidebar to get started!**")


def render_find_staff():
    """Render the find staff feature."""
    st.markdown("## 🔎 Find Staff")
    st.markdown("Take a photo or upload an image to identify a colleague.")

    if not st.session_state.pinecone_helper:
        st.warning("⚠️ Database is not configured. Please add your API key to the `.env` file and restart the app.")
        return

    st.markdown("### Choose Input Method")
    input_method = st.radio(
        "Select how to capture the image:",
        ["📁 Upload Image File", "📸 Use Camera"],
        horizontal=True,
        key="search_input_method",
        label_visibility="collapsed"
    )

    img_path = None

    if input_method == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="search_img", label_visibility="collapsed")
        if uploaded_file:
            img_path = save_uploaded_file(uploaded_file)
    else:
        camera_photo = st.camera_input("Capture image", key="camera_search", label_visibility="collapsed")
        if camera_photo:
            img_path = save_uploaded_file(camera_photo)

    with st.expander("Search Settings"):
        top_k = st.slider("Number of Results", 1, 10, 3)
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)

    if img_path:
        st.markdown("### Uploaded Image")
        display_image_with_info(img_path)

        if st.button("🔍 Find This Person", type="primary"):
            with st.spinner("Searching..."):
                try:
                    embedding, _ = extract_embedding(img_path, model_name=MODEL_NAME)

                    matches = st.session_state.pinecone_helper.search_faces(
                        query_embedding=embedding,
                        top_k=top_k,
                        score_threshold=threshold
                    )

                    st.markdown("---")
                    st.markdown("### Results")

                    if matches:
                        for match in matches:
                            meta = match['metadata']
                            score = match['score']
                            name = meta.get('name', 'Unknown')
                            role = meta.get('role', 'N/A')
                            department = meta.get('department', 'N/A')
                            st.markdown(
                                f"""<div class="staff-card">
                                <h3>{name}</h3>
                                <p><strong>Role:</strong> {role}<br>
                                <strong>Department:</strong> {department}<br>
                                <strong>Match Confidence:</strong> {score:.1%}</p>
                                </div>""",
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("No matches found. This person may not be registered yet.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        if input_method == "📁 Upload Image File":
            st.info("Please upload an image to search.")
        else:
            st.info("Please capture a photo using the camera to search.")


def render_register_staff():
    """Render the staff registration feature."""
    st.markdown("## ➕ Register Staff")
    st.markdown("Register a new staff member by capturing or uploading their photo.")

    if not st.session_state.pinecone_helper:
        st.warning("⚠️ Database is not configured. Please add your API key to the `.env` file and restart the app.")
        return

    st.markdown("### Choose Input Method")
    input_method = st.radio(
        "Select how to capture the face:",
        ["📁 Upload Image File", "📸 Use Camera"],
        horizontal=True,
        label_visibility="collapsed"
    )

    img_path = None

    if input_method == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload Face Image", type=['jpg', 'jpeg', 'png'], key="register_img")
        if uploaded_file:
            img_path = save_uploaded_file(uploaded_file)
    else:
        st.caption("Click the camera button below to capture a photo")
        camera_photo = st.camera_input("Take a picture", key="camera_register")
        if camera_photo:
            img_path = save_uploaded_file(camera_photo)

    if img_path:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Photo")
            display_image_with_info(img_path)

        with col2:
            st.markdown("### Staff Information")

            person_name = st.text_input("Name *", "", placeholder="e.g. Nguyen Van A")
            role = st.text_input("Role *", "", placeholder="e.g. Software Engineer")
            department = st.text_input("Department *", "", placeholder="e.g. Engineering")

            if st.button("➕ Register", type="primary"):
                if not person_name or not role or not department:
                    st.warning("Please fill in all fields (Name, Role, Department).")
                else:
                    with st.spinner("Checking for duplicates..."):
                        try:
                            embedding, _ = extract_embedding(img_path, model_name=MODEL_NAME)

                            # Check if this face is already registered
                            duplicates = st.session_state.pinecone_helper.search_faces(
                                query_embedding=embedding,
                                top_k=1,
                                score_threshold=0.85
                            )

                            if duplicates:
                                match = duplicates[0]
                                meta = match['metadata']
                                existing_name = meta.get('name', 'Unknown')
                                existing_role = meta.get('role', 'N/A')
                                existing_dept = meta.get('department', 'N/A')
                                st.error(
                                    f"This face appears to already be registered as "
                                    f"**{existing_name}** ({existing_role}, {existing_dept}) "
                                    f"with {match['score']:.1%} confidence.\n\n"
                                    f"To update their information, delete the existing entry "
                                    f"from **Staff Directory** first."
                                )
                            else:
                                face_id = f"staff_{uuid.uuid4().hex[:8]}"

                                metadata = {
                                    "name": person_name,
                                    "role": role,
                                    "department": department,
                                    "registered_at": datetime.now().isoformat(),
                                }

                                success = st.session_state.pinecone_helper.register_face(
                                    embedding=embedding,
                                    face_id=face_id,
                                    metadata=metadata
                                )

                                if success:
                                    st.success(f"Registered **{person_name}** successfully!")
                                    st.balloons()

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    else:
        if input_method == "📁 Upload Image File":
            st.info("Please upload a photo to register.")
        else:
            st.info("Please capture a photo using the camera to register.")


def render_staff_directory():
    """Render the staff directory page."""
    st.markdown("## 📋 Staff Directory")
    st.markdown("Browse all registered staff members.")

    if not st.session_state.pinecone_helper:
        st.warning("⚠️ Database is not configured. Please add your API key to the `.env` file and restart the app.")
        return

    if st.button("🔄 Refresh"):
        st.rerun()

    with st.spinner("Loading staff directory..."):
        try:
            entries = st.session_state.pinecone_helper.list_all_faces()

            if not entries:
                st.info("No staff members registered yet. Go to **Register Staff** to add someone.")
                return

            st.markdown(f"**Total: {len(entries)} staff member(s)**")
            st.markdown("---")

            for entry in entries:
                meta = entry.get('metadata', {})
                face_id = entry.get('id', '')
                name = meta.get('name', 'Unknown')
                role = meta.get('role', 'N/A')
                department = meta.get('department', 'N/A')
                registered_at = meta.get('registered_at', 'N/A')

                with st.container():
                    st.markdown(
                        f"**{name}** — {role}, {department}  \n"
                        f"<small>Registered: {registered_at}</small>",
                        unsafe_allow_html=True
                    )
                    if st.button("🗑️ Delete", key=f"del_{face_id}"):
                        try:
                            st.session_state.pinecone_helper.delete_face(face_id)
                            st.success(f"Deleted {name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {str(e)}")
                    st.markdown("---")

        except Exception as e:
            st.error(f"Error loading directory: {str(e)}")


def main():
    """Main application entry point."""
    initialize_session_state()

    # Try to initialize Pinecone
    if not st.session_state.pinecone_initialized:
        try:
            st.session_state.pinecone_helper = initialize_pinecone_from_env()
            st.session_state.pinecone_initialized = True
        except Exception:
            st.session_state.pinecone_helper = None
            st.session_state.pinecone_initialized = True

    render_header()

    feature = render_sidebar()

    if feature == "🏠 Home":
        render_home()
    elif feature == "🔎 Find Staff":
        render_find_staff()
    elif feature == "➕ Register Staff":
        render_register_staff()
    elif feature == "📋 Staff Directory":
        render_staff_directory()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Staff Directory | Powered by DeepFace & Pinecone"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
