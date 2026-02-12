# Face Recognition & Analysis Web Application 👤

A modern, interactive Streamlit web application for face recognition, verification, and facial attribute analysis using DeepFace and Pinecone vector database.

![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Face Verification** - Compare two face images to verify if they belong to the same person (supports file upload & webcam)
- **Facial Attribute Analysis** - Analyze age, gender, emotion, and race from face images (supports file upload & webcam)
- **Face Search** - Search for similar faces in your Pinecone vector database (supports file upload & webcam)
- **Face Registration** - Add new faces to your database with metadata (supports file upload & webcam)
- **Multiple Models** - Support for 9+ state-of-the-art face recognition models
- **Beautiful UI** - Modern, responsive interface built with Streamlit

## 🚀 Quick Start

### Prerequisites

- Python 3.13 or higher
- Pinecone API key (for face search/registration features)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd c:\Development\FacialRegconize
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Pinecone (Optional - required only for search/registration features):**
   
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your Pinecone credentials:
   ```env
   PINECONE_API_KEY=your_actual_api_key_here
   PINECONE_INDEX_NAME=face-recognition-index
   ```
   
   To get a Pinecone API key:
   - Sign up at [https://www.pinecone.io/](https://www.pinecone.io/)
   - Create a new project
   - Copy your API key from the dashboard

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## 📖 Usage Guide

### Face Verification

1. Select "✅ Face Verification" from the sidebar
2. Choose input method (📁 Upload Files or 📸 Use Camera)
3. Provide two face images
4. Click "Verify Faces"
5. View the verification result with confidence score

### Facial Analysis

1. Select "🔍 Facial Analysis" from the sidebar
2. Choose input method (📁 Upload File or 📸 Use Camera)
3. Provide a face image
4. Click "Analyze Face"
5. View detected attributes: age, gender, emotion, race

### Face Search (Requires Pinecone)

1. Select "🔎 Face Search" from the sidebar
2. Choose input method (📁 Upload File or 📸 Use Camera)
3. Provide a query face image
4. Adjust search parameters (top K results, similarity threshold)
5. Click "Search Similar Faces"
6. View matching faces from your database

### Face Registration (Requires Pinecone)

1. Select "➕ Register Face" from the sidebar
2. Choose input method:
   - **📁 Upload Image File**: Select and upload a face image
   - **📸 Use Camera**: Capture a photo directly from your webcam
3. Enter person details (name, notes)
4. Click "Register Face"
5. The face embedding will be stored in Pinecone

## 🔧 Configuration

### Supported Face Recognition Models

- **ArcFace** (Recommended - best accuracy)
- VGG-Face
- Facenet / Facenet512
- OpenFace
- DeepFace
- DeepID
- Dlib
- SFace

You can select the model from the sidebar in the application.

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PINECONE_API_KEY` | Your Pinecone API key | For search/registration |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index | For search/registration |

## 📁 Project Structure

```
c:\Development\FacialRegconize\
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
├── .env                       # Your configuration (create this)
├── .gitignore                 # Git ignore file
├── README.md                  # This file
└── utils/                     # Utility modules
    ├── __init__.py
    ├── deepface_helper.py     # DeepFace wrapper functions
    ├── pinecone_helper.py     # Pinecone integration
    └── image_utils.py         # Image processing utilities
```

## 🔬 Technology Stack

- **Streamlit** - Web application framework
- **DeepFace** - Face recognition and analysis library
- **Pinecone** - Vector database for similarity search
- **OpenCV** - Image processing
- **TensorFlow/Keras** - Deep learning backend

## 🐛 Troubleshooting

### Issue: "No module named 'deepface'"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "Pinecone not connected"
**Solution:** 
1. Make sure you've created a `.env` file
2. Verify your API key is correct
3. Restart the Streamlit application

### Issue: "No face detected"
**Solution:** 
1. Ensure the image has a clear, visible face
2. Try a different image with better lighting
3. Make sure the face is not too small in the image

### Issue: TensorFlow warnings
**Solution:** These are usually harmless warnings about GPU optimization. The app will still work fine on CPU.

## ⚖️ Privacy & Ethics

This application is for educational and demonstration purposes. When using face recognition technology:

- ✅ Always obtain consent before capturing/analyzing faces
- ✅ Ensure compliance with privacy laws (GDPR, CCPA, etc.)
- ✅ Handle biometric data with appropriate security measures
- ✅ Be aware of potential biases in facial recognition models
- ✅ Use responsibly and ethically

## 📚 Resources

- [DeepFace GitHub Repository](https://github.com/serengil/deepface)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **DeepFace** library by [Sefik Ilkin Serengil](https://github.com/serengil)
- **Pinecone** for vector database capabilities
- **Streamlit** for the amazing web framework

---

Built with ❤️ using modern AI and web technologies
