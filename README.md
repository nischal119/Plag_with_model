# Plagiarism Detection System

A comprehensive plagiarism detection system built from scratch using only NumPy, featuring a modern web interface for text similarity analysis and plagiarism detection.

## Features

### Core Algorithm

- **NumPy-only implementation**: No scikit-learn, TensorFlow, or PyTorch dependencies
- **Advanced feature extraction**:
  - Unigram Jaccard similarity
  - Bigram Jaccard similarity
  - Trigram Jaccard similarity
  - Cosine similarity using TF vectorization
- **Logistic Regression model** implemented from scratch using gradient descent
- **Text preprocessing**: Lowercasing, punctuation removal, tokenization

### Web Interface

- **Modern, responsive UI** with beautiful design
- **Three detection modes**:
  - Pairwise comparison between two texts/files
  - Classification mode for single text analysis
  - Multiple reference comparison
- **File upload support** for `.txt`, `.docx`, and `.pdf` files
- **Real-time results** with similarity scores and highlighted matches
- **Visual plagiarism indicators** with color-coded scores

### Technical Features

- **Modular architecture** with clean, well-commented code
- **RESTful API** with Flask backend
- **File processing** for multiple document formats
- **Comprehensive error handling** and user feedback
- **Responsive design** for mobile and desktop

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd Plag_with_model
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to `http://localhost:5000`

## Usage

### 1. Training the Model

- Click the "Train Model" button to train the plagiarism detection model
- The system will use the provided `data.csv` dataset
- Training results will show accuracy metrics

### 2. Pairwise Comparison

- Select "Pairwise Comparison" mode
- Enter text in both text areas or upload files
- Click "Compare Texts" to analyze similarity
- View detailed results with similarity scores and highlighted matches

### 3. Classification Mode

- Select "Classification" mode
- Enter text or upload a file
- Click "Classify Text" to determine if content is likely plagiarized
- View classification probability and similarity metrics

### 4. Multiple Reference Comparison

- Select "Multiple Reference" mode
- Enter text or upload a file
- Click "Compare Against References" to test against multiple reference texts
- View ranked results by similarity score

## Dataset Format

The system expects a CSV file with the following columns:

- `source_txt`: Source text content
- `plagiarism_txt`: Text to compare against source
- `label`: Binary label (1 = plagiarized, 0 = not plagiarized)

## System Architecture

### Core Components

1. **TextPreprocessor** (`plagiarism_detector.py`)

   - Text cleaning and normalization
   - Tokenization and n-gram generation
   - Punctuation removal and lowercasing

2. **FeatureExtractor** (`plagiarism_detector.py`)

   - Jaccard similarity calculations (unigram, bigram, trigram)
   - TF-based cosine similarity
   - Feature vector generation

3. **LogisticRegression** (`plagiarism_detector.py`)

   - Gradient descent implementation
   - Sigmoid activation function
   - Cost function and gradient computation

4. **PlagiarismDetector** (`plagiarism_detector.py`)

   - Main detection pipeline
   - Model training and evaluation
   - Matching phrase detection

5. **Flask Web Application** (`app.py`)
   - RESTful API endpoints
   - File upload handling
   - Session management

### File Structure

```
Plag_with_model/
├── app.py                 # Flask web application
├── plagiarism_detector.py # Core detection algorithm
├── requirements.txt       # Python dependencies
├── data.csv              # Training dataset
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── style.css         # CSS styles
│   └── script.js         # JavaScript functionality
└── uploads/              # Temporary file storage
```

## API Endpoints

- `GET /` - Main web interface
- `POST /train` - Train the plagiarism detection model
- `POST /detect` - Perform plagiarism detection
- `POST /compare_multiple` - Compare against multiple references

## Technical Details

### Feature Extraction

The system extracts 4 key features for each text pair:

1. **Unigram Jaccard Similarity**: Word-level overlap
2. **Bigram Jaccard Similarity**: Two-word phrase overlap
3. **Trigram Jaccard Similarity**: Three-word phrase overlap
4. **Cosine Similarity**: TF-based vector similarity

### Model Training

- **Algorithm**: Logistic Regression with gradient descent
- **Optimization**: Stochastic gradient descent
- **Regularization**: None (can be added for production)
- **Evaluation**: Train/test split with accuracy metrics

### Performance Considerations

- **Memory efficient**: Processes large datasets in chunks
- **Scalable**: Modular design allows for easy extension
- **Fast inference**: Optimized NumPy operations
- **File handling**: Supports multiple document formats

## Customization

### Adding New Features

1. Extend the `FeatureExtractor` class
2. Add new similarity metrics
3. Update the feature vector size in `LogisticRegression`

### Modifying the Model

1. Adjust hyperparameters in `LogisticRegression.__init__()`
2. Implement different optimization algorithms
3. Add regularization or other ML techniques

### UI Customization

1. Modify `templates/index.html` for layout changes
2. Update `static/style.css` for styling
3. Extend `static/script.js` for new functionality

## Limitations and Future Improvements

### Current Limitations

- Basic text preprocessing (no stemming/lemmatization)
- Simple phrase matching algorithm
- Limited to English text
- No semantic similarity analysis

### Potential Improvements

- **Advanced NLP**: Add stemming, lemmatization, POS tagging
- **Semantic Analysis**: Implement word embeddings
- **Deep Learning**: Add neural network models
- **Multi-language Support**: Extend to other languages
- **Performance Optimization**: Add caching and parallel processing
- **Security**: Add input validation and rate limiting

## Troubleshooting

### Common Issues

1. **Model training fails**:

   - Check if `data.csv` exists and has correct format
   - Verify CSV column names match expected format
   - Ensure sufficient memory for large datasets

2. **File upload errors**:

   - Check file format (only .txt, .docx, .pdf supported)
   - Verify file size (max 16MB)
   - Ensure proper file permissions

3. **Web interface not loading**:
   - Check if Flask server is running
   - Verify port 5000 is available
   - Check browser console for JavaScript errors

### Performance Tips

- Use smaller datasets for testing
- Close other applications to free memory
- Consider using a production WSGI server for deployment

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- Built with NumPy for numerical computations
- Flask for web framework
- Font Awesome for icons
- Modern CSS for responsive design
