# MemeAI

## AI-Powered Multimodal Meme Search System

MemeAI is an AI-powered meme search application that allows users to find relevant memes using natural language text or an uploaded meme image.

The system combines Natural Language Processing (NLP), Computer Vision, Machine Learning embeddings, and similarity search to provide semantic and visual meme search.

---

## Features

- Text-based semantic meme search
- Image-based similar meme search
- Natural language query understanding
- Sentence Transformer embeddings
- CLIP-based image embeddings
- Cosine similarity-based ranking
- FastAPI REST API
- React-based frontend
- Meme image preview
- Similarity scores for search results
- Search evaluation using Recall@5 and MRR
- Responsive and user-friendly search interface

---

## How It Works

### Text Search

The text search system converts both meme text and the user's query into numerical embeddings using a Sentence Transformer model.

```text
User Query
     ↓
Sentence Transformer
     ↓
Text Embedding
     ↓
Cosine Similarity
     ↓
Rank Meme Embeddings
     ↓
Top Matching Memes
Image Search

The image search system uses CLIP to convert the uploaded image and stored meme images into image embeddings.

Uploaded Image
     ↓
CLIP Vision Encoder
     ↓
Image Embedding
     ↓
Cosine Similarity
     ↓
Rank Image Embeddings
     ↓
Similar Memes
Technology Stack
Machine Learning
Python
Sentence Transformers
CLIP
PyTorch
NumPy
Pandas
Scikit-learn
Backend
FastAPI
Uvicorn
Python Multipart
Frontend
React
Vite
JavaScript
Axios
CSS
Machine Learning Models
Text Embedding Model

MemeAI uses the:

all-MiniLM-L6-v2

Sentence Transformer model to convert text into 384-dimensional embeddings.

These embeddings are used to perform semantic similarity search between the user's query and meme-related text.

Image Embedding Model

MemeAI uses CLIP for image representation.

The CLIP vision model generates 512-dimensional image embeddings.

These embeddings are compared using cosine similarity to identify visually similar memes.

Dataset

The project uses a meme dataset containing approximately 7,000 meme images with associated text/OCR information and sentiment labels.

The dataset is processed before being used by the search system.

The preprocessing pipeline creates structured searchable information containing:

Meme image
Caption
OCR text
Sentiment
Search text
Data Processing Pipeline
Raw Dataset
     ↓
Dataset Cleaning
     ↓
Text Preparation
     ↓
Image Processing
     ↓
Text Embedding Generation
     ↓
Image Embedding Generation
     ↓
Searchable Embedding Database
Text Search Pipeline
User enters a query
        ↓
Query converted into embedding
        ↓
Compare with meme text embeddings
        ↓
Calculate cosine similarity
        ↓
Sort results by similarity
        ↓
Display top matching memes

Example queries:

exam tomorrow and I have not studied
college assignment deadline
feeling angry at someone
I finished my project successfully
Image Search Pipeline
User uploads an image
        ↓
Image converted to RGB
        ↓
CLIP image embedding generated
        ↓
Compare with stored image embeddings
        ↓
Calculate cosine similarity
        ↓
Rank visually similar memes
        ↓
Display similar memes
Project Structure
MemeAI/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   └── package.json
│
├── ml/
│
i mena everything from top to tech stack and all everything

Yes — you mean the entire README from the very first line all the way to the end, in one single copy-paste block, including Tech Stack, Features, Installation, Structure, API, ML models, evaluation, future scope, etc.

Copy everything inside this block into README.md:

# MemeAI

## AI-Powered Multimodal Meme Search System

MemeAI is an AI-powered meme search application that allows users to find relevant memes using natural language text or an uploaded meme image.

The system combines Natural Language Processing (NLP), Computer Vision, Machine Learning embeddings, and similarity search to provide semantic and visual meme search.

---

## Features

- Text-based semantic meme search
- Image-based similar meme search
- Natural language query understanding
- Sentence Transformer embeddings for text
- CLIP-based image embeddings
- Cosine similarity-based ranking
- FastAPI REST API
- React-based frontend
- Meme image preview
- Similarity scores for search results
- Search evaluation using Recall@5 and MRR
- Interactive and user-friendly search interface

---

## How It Works

MemeAI supports two main search methods:

1. Text-based meme search
2. Image-based similar meme search

### Text Search

The text search system converts the user's query and meme-related text into numerical embeddings using a Sentence Transformer model.

```text
User Query
     ↓
Sentence Transformer
     ↓
Text Embedding
     ↓
Cosine Similarity
     ↓
Rank Meme Embeddings
     ↓
Top Matching Memes
Image Search

The image search system uses CLIP to convert the uploaded image and stored meme images into image embeddings.

Uploaded Image
     ↓
CLIP Vision Encoder
     ↓
Image Embedding
     ↓
Cosine Similarity
     ↓
Rank Image Embeddings
     ↓
Visually Similar Memes
Technology Stack
Machine Learning
Python
Sentence Transformers
CLIP
PyTorch
NumPy
Pandas
Scikit-learn
Pillow
Backend
FastAPI
Uvicorn
Python Multipart
Frontend
React
Vite
JavaScript
Axios
CSS
Development Tools
Git
GitHub
Visual Studio Code
Python Virtual Environment
Machine Learning Models
Text Embedding Model

MemeAI uses the:

all-MiniLM-L6-v2

Sentence Transformer model.

The model converts text into 384-dimensional numerical embeddings.

These embeddings are used to perform semantic similarity search between the user's query and the meme dataset.

Image Embedding Model

MemeAI uses CLIP for image representation.

The CLIP vision model generates 512-dimensional image embeddings.

These embeddings are compared using cosine similarity to identify visually similar memes.

Dataset

The project uses a meme dataset containing approximately 7,000 meme images along with associated text/OCR information and sentiment labels.

The dataset contains information such as:

Meme image
Caption
OCR text
Sentiment

The dataset is cleaned and transformed before generating embeddings.

Data Processing Pipeline
Raw Dataset
     ↓
Dataset Cleaning
     ↓
Text Preparation
     ↓
Image Processing
     ↓
Text Embedding Generation
     ↓
Image Embedding Generation
     ↓
Searchable Embedding Data
Text Search Pipeline
User enters a query
        ↓
Query converted into embedding
        ↓
Compare with meme text embeddings
        ↓
Calculate cosine similarity
        ↓
Sort results by similarity
        ↓
Display top matching memes
Example Queries
exam tomorrow and I have not studied
college assignment deadline
feeling angry at someone
I finished my project successfully
everyone is studying except me
Image Search Pipeline
User uploads an image
        ↓
Image converted to RGB
        ↓
CLIP image embedding generated
        ↓
Compare with stored image embeddings
        ↓
Calculate cosine similarity
        ↓
Rank visually similar memes
        ↓
Display similar memes

The image search system focuses on visual similarity between the uploaded image and memes in the dataset.

Project Structure
MemeAI/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── vite.config.js
│
├── ml/
│   ├── prepare_dataset.py
│   ├── create_embeddings.py
│   ├── create_image_embeddings.py
│   ├── search_meme.py
│   ├── search_image.py
│   ├── evaluate_search.py
│   ├── evaluate_recall.py
│   └── view_results.py
│
├── dataset/
│   ├── memes_real.csv
│   ├── evaluation_queries.csv
│   ├── meme_embeddings.npy
│   └── image_embeddings.npy
│
├── images/
│   └── images/
│
├── labels.csv
├── .gitignore
└── README.md
Description of Important Files
Backend
backend/main.py

Contains the FastAPI application and API endpoints for text and image-based meme search.

Frontend
frontend/src/App.jsx

Contains the main React application and search interface.

frontend/src/App.css

Contains the styling for the application.

frontend/src/index.css

Contains global CSS settings.

Machine Learning
ml/prepare_dataset.py

Prepares and cleans the dataset for the search system.

ml/create_embeddings.py

Generates text embeddings using Sentence Transformers.

ml/create_image_embeddings.py

Generates image embeddings using CLIP.

ml/search_meme.py

Performs text-based meme search from the command line.

ml/search_image.py

Performs image-based similar meme search from the command line.

ml/evaluate_search.py

Evaluates the text search system.

ml/evaluate_recall.py

Calculates search evaluation metrics such as Recall@5 and MRR.

Installation
1. Clone the Repository
git clone <your-repository-url>
cd MemeAI
2. Create a Python Virtual Environment
python -m venv venv
3. Activate the Virtual Environment

On Windows:

venv\Scripts\activate
4. Install Python Dependencies
pip install sentence-transformers pandas scikit-learn fastapi uvicorn pillow transformers torch torchvision python-multipart
5. Install Frontend Dependencies
cd frontend
npm install
Running the Project

The backend and frontend should be run in separate terminals.

Start the Backend

From the project root:

cd D:\MemeAI
venv\Scripts\activate
python -m uvicorn backend.main:app --reload

The backend will run at:

http://127.0.0.1:8000
Start the Frontend

Open another terminal:

cd D:\MemeAI\frontend
npm run dev

The frontend will normally run at:

http://localhost:5173

Open the frontend URL in a browser to use MemeAI.

API Endpoints
Health Check
GET /health

Used to verify that the backend is running correctly.

Text Search
GET /api/search

Example:

/api/search?query=exam tomorrow and I have not studied

The endpoint returns memes ranked according to semantic similarity.

Image Search
POST /api/image-search

The endpoint accepts an uploaded image and returns visually similar memes.

Search Results

Each result can contain information such as:

Meme image
Caption
Sentiment
Similarity score
Image URL

The results are ranked from highest to lowest similarity.

Similarity Calculation

MemeAI uses cosine similarity to compare embeddings.

For text search:

User Query Embedding
        ↕
Meme Text Embedding

For image search:

Uploaded Image Embedding
        ↕
Stored Meme Image Embedding

A higher similarity value indicates that the embeddings are more similar according to the respective model representation.

Search Evaluation

The project includes evaluation scripts to measure the performance of the semantic search system.

The evaluation includes:

Top-1 accuracy
Recall@5
Mean Reciprocal Rank (MRR)

The evaluation queries contain example search scenarios such as:

exam tomorrow and I have not studied
college assignment deadline
feeling angry at someone
feeling happy after success
college life is difficult

The current evaluation uses keyword matching as a simple proxy for relevant results rather than a fully human-labeled ground-truth dataset.

Current Scope

MemeAI currently supports:

1. Text → Meme Search

Users can enter a natural language query and retrieve semantically related memes.

2. Image → Similar Meme Search

Users can upload an image and retrieve visually similar memes from the dataset.

Current Limitations

The current image search system provides visual similarity.

It does not guarantee:

Exact movie identification
Exact actor identification
Original meme source identification
Exact meme template identification

These features would require additional labeled data and specialized models.

Future Improvements

Possible future improvements include:

Voice → Text → Meme Search
Exact meme source identification
Movie and actor recognition
Multimodal text + image ranking
FAISS-based vector search
PostgreSQL with pgvector
User accounts
Search history
Personalized recommendations
Improved human-labeled evaluation dataset
Advanced recommendation and ranking
Scalable cloud deployment
Learning Outcomes

This project provides practical experience with:

Artificial Intelligence
Machine Learning
Natural Language Processing
Computer Vision
Transfer Learning
Text Embeddings
Image Embeddings
Semantic Search
Image Similarity
Cosine Similarity
REST API development
FastAPI
React
Machine Learning evaluation
Git and GitHub
Why MemeAI?

Traditional keyword-based meme search may fail when the user's query does not exactly match the text associated with a meme.

MemeAI addresses this by using embeddings to represent the meaning and visual characteristics of memes.

This allows the system to perform similarity-based search rather than relying only on exact keyword matching.

Example Workflow

A user can search for:

"I have an exam tomorrow and I haven't studied"

The system converts the query into an embedding and compares it with the stored meme text embeddings.

The most similar memes are then ranked and displayed in the frontend.

Alternatively, the user can upload an image.

The system generates a CLIP embedding for the uploaded image, compares it with stored meme image embeddings, and displays visually similar memes.

Application Architecture
                    MEMEAI
                       │
          ┌────────────┴────────────┐
          │                         │
     Text Search              Image Search
          │                         │
          ↓                         ↓
 Sentence Transformer             CLIP
          │                         │
          ↓                         ↓
   Text Embeddings          Image Embeddings
          │                         │
          └────────────┬────────────┘
                       ↓
               Similarity Search
                       ↓
                 FastAPI Backend
                       ↓
                  React Frontend
                       ↓
                 Search Results
Security and Repository Management

The project uses a .gitignore file to avoid committing unnecessary or generated files such as:

Python virtual environment
Generated embeddings
Model files
Environment variables
Cache files
IDE configuration files

Large datasets and generated files should be managed separately when necessary.