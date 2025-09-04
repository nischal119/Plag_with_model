import math
import re
import string
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np


class TextPreprocessor:
    """Text preprocessing utilities for plagiarism detection."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by lowercasing and removing punctuation."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        return text.split()

    @staticmethod
    def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Generate n-grams from tokens."""
        if n == 1:
            return [(token,) for token in tokens]
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        """Complete text preprocessing pipeline."""
        cleaned = TextPreprocessor.clean_text(text)
        tokens = TextPreprocessor.tokenize(cleaned)
        return tokens


class FeatureExtractor:
    """Feature extraction for plagiarism detection."""

    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def extract_features(text1: str, text2: str) -> np.ndarray:
        """Extract all features for a pair of texts."""
        # Preprocess texts
        tokens1 = TextPreprocessor.preprocess_text(text1)
        tokens2 = TextPreprocessor.preprocess_text(text2)

        # Generate n-grams
        unigrams1 = set(TextPreprocessor.get_ngrams(tokens1, 1))
        unigrams2 = set(TextPreprocessor.get_ngrams(tokens2, 1))

        bigrams1 = set(TextPreprocessor.get_ngrams(tokens1, 2))
        bigrams2 = set(TextPreprocessor.get_ngrams(tokens2, 2))

        trigrams1 = set(TextPreprocessor.get_ngrams(tokens1, 3))
        trigrams2 = set(TextPreprocessor.get_ngrams(tokens2, 3))

        # Calculate Jaccard similarities
        unigram_jaccard = FeatureExtractor.jaccard_similarity(unigrams1, unigrams2)
        bigram_jaccard = FeatureExtractor.jaccard_similarity(bigrams1, bigrams2)
        trigram_jaccard = FeatureExtractor.jaccard_similarity(trigrams1, trigrams2)

        # Calculate TF-based cosine similarity
        cosine_sim = FeatureExtractor._calculate_tf_cosine_similarity(tokens1, tokens2)

        return np.array([unigram_jaccard, bigram_jaccard, trigram_jaccard, cosine_sim])

    @staticmethod
    def _calculate_tf_cosine_similarity(
        tokens1: List[str], tokens2: List[str]
    ) -> float:
        """Calculate cosine similarity using TF vectorization."""
        # Get all unique words
        all_words = list(set(tokens1 + tokens2))
        word_to_idx = {word: idx for idx, word in enumerate(all_words)}

        # Create TF vectors
        tf1 = np.zeros(len(all_words))
        tf2 = np.zeros(len(all_words))

        # Count frequencies
        for token in tokens1:
            if token in word_to_idx:
                tf1[word_to_idx[token]] += 1

        for token in tokens2:
            if token in word_to_idx:
                tf2[word_to_idx[token]] += 1

        # Normalize by document length
        if len(tokens1) > 0:
            tf1 = tf1 / len(tokens1)
        if len(tokens2) > 0:
            tf2 = tf2 / len(tokens2)

        return FeatureExtractor.cosine_similarity(tf1, tf2)


class LogisticRegression:
    """Logistic Regression implementation using only NumPy."""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.costs = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute logistic regression cost."""
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)

        # Avoid log(0)
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)

        cost = -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def compute_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Compute gradients for logistic regression."""
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)

        dw = (1 / m) * np.dot(X.T, (h - y))
        db = (1 / m) * np.sum(h - y)

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the logistic regression model."""
        # Ensure inputs have correct shapes
        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array of shape (num_samples, num_features)"
            )

        num_samples, num_features = X.shape
        y = y.reshape(-1)  # ensure shape (num_samples,)

        # Initialize parameters as 1D weight vector (num_features,)
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # Gradient descent
        for i in range(self.max_iterations):
            # Compute gradients
            dw, db = self.compute_gradients(X, y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute and store cost
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                self.costs.append(cost)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save_model(self, filename: str) -> None:
        """Save model weights and bias to file."""
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be trained before saving")
        np.savez(filename, weights=self.weights, bias=self.bias)

    def load_model(self, filename: str) -> bool:
        """Load model weights and bias from file."""
        try:
            data = np.load(filename)
            self.weights = data["weights"]
            self.bias = data["bias"]
            return True
        except (FileNotFoundError, KeyError, OSError):
            return False

    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.weights is not None and self.bias is not None


class PlagiarismDetector:
    """Main plagiarism detection system."""

    def __init__(self, model_file: str = "plagiarism_model.npz"):
        self.model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
        self.feature_extractor = FeatureExtractor()
        self.model_file = model_file
        self.is_trained = False
        self._load_existing_model()

    def _load_existing_model(self) -> None:
        """Try to load an existing trained model."""
        if self.model.load_model(self.model_file):
            self.is_trained = True
            print(f"Loaded existing model from {self.model_file}")
        else:
            print(f"No existing model found at {self.model_file}. Training required.")

    def prepare_dataset(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset from CSV file."""
        import pandas as pd

        # Load data
        df = pd.read_csv(data_path)

        # Extract features for all pairs
        features = []
        labels = []

        for _, row in df.iterrows():
            text1 = str(row["source_txt"])
            text2 = str(row["plagiarism_txt"])
            label = int(row["label"])

            # Extract features
            feature_vector = self.feature_extractor.extract_features(text1, text2)
            features.append(feature_vector)
            labels.append(label)

        return np.array(features), np.array(labels)

    def train(self, data_path: str, test_size: float = 0.2) -> Dict[str, float]:
        """Train the plagiarism detection model."""
        # Prepare dataset
        X, y = self.prepare_dataset(data_path)

        # Split into train and test sets
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Save the trained model
        try:
            self.model.save_model(self.model_file)
            print(f"Model saved to {self.model_file}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")

        # Evaluate
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)

        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "model_saved": True,
        }

    def detect_plagiarism(self, text1: str, text2: str) -> Dict[str, any]:
        """Detect plagiarism between two texts."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Extract features
        features = self.feature_extractor.extract_features(text1, text2)

        # Make prediction
        probability = self.model.predict_proba(features.reshape(1, -1))[0]
        prediction = self.model.predict(features.reshape(1, -1))[0]

        # Extract similarity scores
        unigram_jaccard = features[0]
        bigram_jaccard = features[1]
        trigram_jaccard = features[2]
        cosine_sim = features[3]

        return {
            "is_plagiarized": bool(prediction),
            "plagiarism_probability": float(probability),
            "similarity_scores": {
                "unigram_jaccard": float(unigram_jaccard),
                "bigram_jaccard": float(bigram_jaccard),
                "trigram_jaccard": float(trigram_jaccard),
                "cosine_similarity": float(cosine_sim),
            },
            "features": features.tolist(),
        }

    def find_similar_phrases(
        self, tokens1: List[str], tokens2: List[str], min_length: int = 2
    ) -> List[Dict]:
        """Find similar phrases using word-level similarity with improved filtering."""
        similar_phrases = []

        # Create word frequency dictionaries
        word_freq1 = Counter(tokens1)
        word_freq2 = Counter(tokens2)

        # Find common words
        common_words = set(word_freq1.keys()) & set(word_freq2.keys())

        # Track used phrases to avoid duplicates
        used_phrases1 = set()
        used_phrases2 = set()

        # Find phrases with similar word patterns, starting from longer phrases
        for n in range(min(len(tokens1), len(tokens2)), min_length - 1, -1):
            ngrams1 = TextPreprocessor.get_ngrams(tokens1, n)
            ngrams2 = TextPreprocessor.get_ngrams(tokens2, n)

            for ngram1 in ngrams1:
                for ngram2 in ngrams2:
                    # Calculate word similarity
                    words1 = set(ngram1)
                    words2 = set(ngram2)

                    # Jaccard similarity
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)

                    if union > 0:
                        similarity = intersection / union

                        # More selective threshold for semantic matches
                        if similarity >= 0.4:  # Increased threshold to 40%
                            phrase1 = " ".join(ngram1)
                            phrase2 = " ".join(ngram2)

                            # Check if these phrases are already used or are too similar to existing ones
                            if (
                                phrase1 not in used_phrases1
                                and phrase2 not in used_phrases2
                                and phrase1 != phrase2
                            ):

                                # Check if this phrase is not a substring of already used phrases
                                is_substring = False
                                for used_phrase in used_phrases1:
                                    if phrase1 in used_phrase or used_phrase in phrase1:
                                        is_substring = True
                                        break

                                for used_phrase in used_phrases2:
                                    if phrase2 in used_phrase or used_phrase in phrase2:
                                        is_substring = True
                                        break

                                if not is_substring:
                                    similar_phrases.append(
                                        {
                                            "phrase1": phrase1,
                                            "phrase2": phrase2,
                                            "length": n,
                                            "similarity": similarity,
                                            "match_type": "semantic",
                                        }
                                    )

                                    # Mark these phrases as used
                                    used_phrases1.add(phrase1)
                                    used_phrases2.add(phrase2)

        # Sort by similarity and length, then take only the best matches
        similar_phrases.sort(key=lambda x: (x["similarity"], x["length"]), reverse=True)

        # Limit the number of semantic matches to avoid overwhelming results
        return similar_phrases[:10]  # Return only top 10 semantic matches

    def find_matching_phrases(
        self, text1: str, text2: str, min_length: int = 3
    ) -> List[Dict[str, str]]:
        """Find matching phrases between two texts with enhanced detection and deduplication."""
        tokens1 = TextPreprocessor.preprocess_text(text1)
        tokens2 = TextPreprocessor.preprocess_text(text2)

        matches = []
        exact_matches = []
        semantic_matches = []

        # Find common n-grams (exact matches)
        for n in range(min_length, min(len(tokens1), len(tokens2)) + 1):
            ngrams1 = TextPreprocessor.get_ngrams(tokens1, n)
            ngrams2 = TextPreprocessor.get_ngrams(tokens2, n)

            # Find common n-grams
            common = set(ngrams1).intersection(set(ngrams2))

            for ngram in common:
                phrase = " ".join(ngram)
                exact_matches.append(
                    {
                        "phrase": phrase,
                        "length": n,
                        "type": f"exact_{n}-gram",
                        "match_type": "exact",
                        "similarity": 1.0,
                    }
                )

        # Find similar phrases using the improved function
        similar_phrases = self.find_similar_phrases(tokens1, tokens2, min_length=2)

        for phrase_info in similar_phrases:
            # Check if this is not already an exact match
            is_exact = any(
                match["phrase"] == phrase_info["phrase1"]
                or match["phrase"] == phrase_info["phrase2"]
                for match in exact_matches
            )

            if not is_exact:
                semantic_matches.append(
                    {
                        "phrase1": phrase_info["phrase1"],
                        "phrase2": phrase_info["phrase2"],
                        "length": phrase_info["length"],
                        "type": f"semantic_{phrase_info['length']}-gram",
                        "match_type": "semantic",
                        "similarity": phrase_info["similarity"],
                    }
                )

        # Find individual word matches only for significant words (longer than 3 chars)
        words1 = set(tokens1)
        words2 = set(tokens2)
        common_words = words1.intersection(words2)

        # Only add significant individual word matches
        significant_words = [word for word in common_words if len(word) > 3]

        # Limit individual word matches to avoid spam
        for word in significant_words[:5]:  # Only top 5 significant words
            semantic_matches.append(
                {
                    "phrase1": word,
                    "phrase2": word,
                    "length": 1,
                    "type": "semantic_word",
                    "match_type": "semantic",
                    "similarity": 1.0,
                }
            )

        # Combine and sort matches
        matches = exact_matches + semantic_matches
        matches.sort(key=lambda x: (x["length"], x["similarity"]), reverse=True)

        # Return a reasonable number of matches to avoid overwhelming the UI
        return matches[:15]  # Return top 15 matches

    def get_matching_context(
        self, text1: str, text2: str, phrase: str
    ) -> Dict[str, str]:
        """Get context around matching phrases."""
        # Find the phrase in both texts and extract surrounding context
        context_size = 50  # characters before and after

        def find_context(text, phrase):
            # Case-insensitive search
            import re

            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            match = pattern.search(text)

            if not match:
                return phrase, ""

            start = match.start()
            # Extract context
            context_start = max(0, start - context_size)
            context_end = min(len(text), start + len(phrase) + context_size)

            context = text[context_start:context_end]
            return phrase, context

        phrase1, context1 = find_context(text1, phrase)
        phrase2, context2 = find_context(text2, phrase)

        return {
            "phrase": phrase,
            "context1": context1,
            "context2": context2,
            "highlight1": phrase1,
            "highlight2": phrase2,
        }

    def get_model_status(self) -> Dict[str, any]:
        """Get current model status."""
        return {
            "is_trained": self.is_trained,
            "model_file": self.model_file,
            "model_exists": self.model.is_trained(),
        }

    def reset_model(self) -> None:
        """Reset the model and remove saved file."""
        self.is_trained = False
        self.model.weights = None
        self.model.bias = None
        try:
            import os

            if os.path.exists(self.model_file):
                os.remove(self.model_file)
                print(f"Removed model file: {self.model_file}")
        except Exception as e:
            print(f"Warning: Could not remove model file: {e}")


if __name__ == "__main__":
    # Example usage
    detector = PlagiarismDetector()

    # Train the model
    print("Training plagiarism detection model...")
    results = detector.train("data.csv")
    print(f"Training completed!")
    print(f"Train accuracy: {results['train_accuracy']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")

    # Test with sample texts
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A quick brown fox leaps over a lazy dog."

    result = detector.detect_plagiarism(text1, text2)
    print(f"\nPlagiarism detection result:")
    print(f"Is plagiarized: {result['is_plagiarized']}")
    print(f"Probability: {result['plagiarism_probability']:.4f}")
    print(f"Similarity scores: {result['similarity_scores']}")
