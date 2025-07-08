// Global variables
let currentMode = "pairwise";

// DOM elements
const modeBtns = document.querySelectorAll(".mode-btn");
const modeContents = document.querySelectorAll(".mode-content");
const trainBtn = document.getElementById("trainBtn");
const resetBtn = document.getElementById("resetBtn");
const compareBtn = document.getElementById("compareBtn");
const classifyBtn = document.getElementById("classifyBtn");
const multipleBtn = document.getElementById("multipleBtn");
const trainingStatus = document.getElementById("trainingStatus");
const modelStatusText = document.getElementById("modelStatusText");
const modelStatusIcon = document.getElementById("modelStatusIcon");
const resultsSection = document.getElementById("resultsSection");
const resultsContent = document.getElementById("resultsContent");
const loadingOverlay = document.getElementById("loadingOverlay");

// File input elements
const file1 = document.getElementById("file1");
const file2 = document.getElementById("file2");
const classifyFile = document.getElementById("classifyFile");
const multipleFile = document.getElementById("multipleFile");

// File name display elements
const file1Name = document.getElementById("file1Name");
const file2Name = document.getElementById("file2Name");
const classifyFileName = document.getElementById("classifyFileName");
const multipleFileName = document.getElementById("multipleFileName");

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
  setupEventListeners();
  setupFileUploads();
  checkModelStatus();
});

function setupEventListeners() {
  // Mode switching
  modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  });

  // Model management
  trainBtn.addEventListener("click", trainModel);
  resetBtn.addEventListener("click", resetModel);

  // Detection buttons
  compareBtn.addEventListener("click", () => detectPlagiarism("pairwise"));
  classifyBtn.addEventListener("click", () =>
    detectPlagiarism("classification")
  );
  multipleBtn.addEventListener("click", compareMultiple);
}

function setupFileUploads() {
  // File upload handlers
  file1.addEventListener("change", (e) => handleFileUpload(e, file1Name));
  file2.addEventListener("change", (e) => handleFileUpload(e, file2Name));
  classifyFile.addEventListener("change", (e) =>
    handleFileUpload(e, classifyFileName)
  );
  multipleFile.addEventListener("change", (e) =>
    handleFileUpload(e, multipleFileName)
  );
}

function switchMode(mode) {
  // Update active mode button
  modeBtns.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });

  // Update active mode content
  modeContents.forEach((content) => {
    content.classList.toggle("active", content.id === `${mode}Mode`);
  });

  currentMode = mode;
  hideResults();
}

function handleFileUpload(event, nameElement) {
  const file = event.target.files[0];
  if (file) {
    nameElement.textContent = file.name;
    nameElement.style.color = "#38a169";
  } else {
    nameElement.textContent = "";
  }
}

function showLoading() {
  loadingOverlay.style.display = "flex";
}

function hideLoading() {
  loadingOverlay.style.display = "none";
}

function showStatus(message, type = "info") {
  trainingStatus.textContent = message;
  trainingStatus.className = `status ${type}`;
}

function hideResults() {
  resultsSection.style.display = "none";
}

function showResults(content) {
  resultsContent.innerHTML = content;
  resultsSection.style.display = "block";
  resultsSection.scrollIntoView({ behavior: "smooth" });
}

async function checkModelStatus() {
  try {
    const response = await fetch("/model/status");
    const data = await response.json();

    if (data.success) {
      updateModelStatusUI(data.status);
    }
  } catch (error) {
    console.error("Error checking model status:", error);
    updateModelStatusUI({ is_trained: false, model_exists: false });
  }
}

function updateModelStatusUI(status) {
  if (status.is_trained && status.model_exists) {
    modelStatusText.textContent = "Model ready - trained and loaded";
    modelStatusIcon.className = "status-icon ready";
    trainBtn.textContent = "Retrain Model";
  } else {
    modelStatusText.textContent = "Model not trained - training required";
    modelStatusIcon.className = "status-icon not-ready";
    trainBtn.textContent = "Train Model";
  }
}

async function trainModel() {
  showLoading();
  trainBtn.disabled = true;
  showStatus("Training model...", "info");

  try {
    const response = await fetch("/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (data.success) {
      showStatus(
        `Model trained successfully! Train accuracy: ${(
          data.results.train_accuracy * 100
        ).toFixed(2)}%, Test accuracy: ${(
          data.results.test_accuracy * 100
        ).toFixed(2)}%`,
        "success"
      );
      // Update model status after training
      await checkModelStatus();
    } else {
      showStatus(data.message, "error");
    }
  } catch (error) {
    showStatus("Error training model: " + error.message, "error");
  } finally {
    hideLoading();
    trainBtn.disabled = false;
  }
}

async function resetModel() {
  if (
    !confirm(
      "Are you sure you want to reset the model? This will require retraining."
    )
  ) {
    return;
  }

  showLoading();
  resetBtn.disabled = true;

  try {
    const response = await fetch("/model/reset", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (data.success) {
      showStatus("Model reset successfully. Training required.", "info");
      updateModelStatusUI({ is_trained: false, model_exists: false });
    } else {
      showStatus(data.message, "error");
    }
  } catch (error) {
    showStatus("Error resetting model: " + error.message, "error");
  } finally {
    hideLoading();
    resetBtn.disabled = false;
  }
}

async function detectPlagiarism(mode) {
  showLoading();

  try {
    const formData = new FormData();
    formData.append("mode", mode);

    if (mode === "pairwise") {
      const text1 = document.getElementById("text1").value;
      const text2 = document.getElementById("text2").value;

      if (!text1 && !file1.files[0]) {
        alert("Please provide text or file for Text 1");
        hideLoading();
        return;
      }
      if (!text2 && !file2.files[0]) {
        alert("Please provide text or file for Text 2");
        hideLoading();
        return;
      }

      formData.append("text1", text1);
      formData.append("text2", text2);

      if (file1.files[0]) {
        formData.append("file1", file1.files[0]);
      }
      if (file2.files[0]) {
        formData.append("file2", file2.files[0]);
      }
    } else if (mode === "classification") {
      const text = document.getElementById("classifyText").value;

      if (!text && !classifyFile.files[0]) {
        alert("Please provide text or file for classification");
        hideLoading();
        return;
      }

      formData.append("text", text);

      if (classifyFile.files[0]) {
        formData.append("file", classifyFile.files[0]);
      }
    }

    const response = await fetch("/detect", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      displayResults(data, mode);
    } else {
      alert(data.message);
    }
  } catch (error) {
    alert("Error during detection: " + error.message);
  } finally {
    hideLoading();
  }
}

async function compareMultiple() {
  showLoading();

  try {
    const formData = new FormData();
    const text = document.getElementById("multipleText").value;

    if (!text && !multipleFile.files[0]) {
      alert("Please provide text or file for comparison");
      hideLoading();
      return;
    }

    formData.append("text", text);

    if (multipleFile.files[0]) {
      formData.append("file", multipleFile.files[0]);
    }

    const response = await fetch("/compare_multiple", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      displayMultipleResults(data);
    } else {
      alert(data.message);
    }
  } catch (error) {
    alert("Error during multiple comparison: " + error.message);
  } finally {
    hideLoading();
  }
}

function displayResults(data, mode) {
  let content = "";

  if (mode === "pairwise") {
    content = `
            <div class="result-card">
                <div class="result-header">
                    <div class="result-title">Plagiarism Detection Result</div>
                    <div class="result-score ${getScoreClass(
                      data.result.plagiarism_probability
                    )}">
                        ${(data.result.plagiarism_probability * 100).toFixed(
                          1
                        )}%
                    </div>
                </div>
                
                <div class="similarity-scores">
                    <div class="similarity-item">
                        <div class="similarity-label">Unigram Jaccard</div>
                        <div class="similarity-value">${(
                          data.result.similarity_scores.unigram_jaccard * 100
                        ).toFixed(2)}%</div>
                    </div>
                    <div class="similarity-item">
                        <div class="similarity-label">Bigram Jaccard</div>
                        <div class="similarity-value">${(
                          data.result.similarity_scores.bigram_jaccard * 100
                        ).toFixed(2)}%</div>
                    </div>
                    <div class="similarity-item">
                        <div class="similarity-label">Trigram Jaccard</div>
                        <div class="similarity-value">${(
                          data.result.similarity_scores.trigram_jaccard * 100
                        ).toFixed(2)}%</div>
                    </div>
                    <div class="similarity-item">
                        <div class="similarity-label">Cosine Similarity</div>
                        <div class="similarity-value">${(
                          data.result.similarity_scores.cosine_similarity * 100
                        ).toFixed(2)}%</div>
                    </div>
                </div>

                <div class="matching-phrases">
                    <h4>Matching Phrases:</h4>
                    ${data.matches
                      .map(
                        (match) => `
                        <div class="phrase-item">
                            <div class="phrase-text">"${match.phrase}"</div>
                            <div class="phrase-type">${match.type}</div>
                        </div>
                    `
                      )
                      .join("")}
                </div>

                <div class="highlighted-text">
                    <h4>Text 1 (with highlights):</h4>
                    <div>${data.highlighted_text1}</div>
                </div>

                <div class="highlighted-text">
                    <h4>Text 2 (with highlights):</h4>
                    <div>${data.highlighted_text2}</div>
                </div>
            </div>
        `;
  } else if (mode === "classification") {
    content = `
            <div class="result-card">
                <div class="result-header">
                    <div class="result-title">Classification Result</div>
                    <div class="result-score ${getScoreClass(
                      data.result.plagiarism_probability
                    )}">
                        ${(data.result.plagiarism_probability * 100).toFixed(
                          1
                        )}%
                    </div>
                </div>
                
                <p><strong>Classification:</strong> ${
                  data.result.is_plagiarized
                    ? "Likely Plagiarized"
                    : "Likely Original"
                }</p>
                
                <div class="similarity-scores">
                    <div class="similarity-item">
                        <div class="similarity-label">Unigram Jaccard</div>
                        <div class="similarity-value">${(
                          data.result.similarity_scores.unigram_jaccard * 100
                        ).toFixed(2)}%</div>
                    </div>
                    <div class="similarity-item">
                        <div class="similarity-label">Bigram Jaccard</div>
                        <div class="similarity-value">${(
                          data.result.similarity_scores.bigram_jaccard * 100
                        ).toFixed(2)}%</div>
                    </div>
                    <div class="similarity-item">
                        <div class="similarity-label">Trigram Jaccard</div>
                        <div class="similarity-value">${(
                          data.result.similarity_scores.trigram_jaccard * 100
                        ).toFixed(2)}%</div>
                    </div>
                    <div class="similarity-item">
                        <div class="similarity-label">Cosine Similarity</div>
                        <div class="similarity-value">${(
                          data.result.similarity_scores.cosine_similarity * 100
                        ).toFixed(2)}%</div>
                    </div>
                </div>
            </div>
        `;
  }

  showResults(content);
}

function displayMultipleResults(data) {
  let content = `
        <div class="result-card">
            <h3>Multiple Reference Comparison Results</h3>
            <p><strong>Input Text:</strong> ${data.input_text.substring(
              0,
              100
            )}${data.input_text.length > 100 ? "..." : ""}</p>
        </div>
    `;

  data.results.forEach((item, index) => {
    content += `
            <div class="result-card">
                <div class="result-header">
                    <div class="result-title">Reference ${
                      item.reference_id
                    }</div>
                    <div class="result-score ${getScoreClass(
                      item.result.plagiarism_probability
                    )}">
                        ${(item.result.plagiarism_probability * 100).toFixed(
                          1
                        )}%
                    </div>
                </div>
                
                <p><strong>Reference Text:</strong> ${item.reference_text}</p>
                <p><strong>Classification:</strong> ${
                  item.result.is_plagiarized
                    ? "Likely Plagiarized"
                    : "Likely Original"
                }</p>
                
                <div class="similarity-scores">
                    <div class="similarity-item">
                        <div class="similarity-label">Unigram Jaccard</div>
                        <div class="similarity-value">${(
                          item.result.similarity_scores.unigram_jaccard * 100
                        ).toFixed(2)}%</div>
                    </div>
                    <div class="similarity-item">
                        <div class="similarity-label">Bigram Jaccard</div>
                        <div class="similarity-value">${(
                          item.result.similarity_scores.bigram_jaccard * 100
                        ).toFixed(2)}%</div>
                    </div>
                    <div class="similarity-item">
                        <div class="similarity-label">Trigram Jaccard</div>
                        <div class="similarity-value">${(
                          item.result.similarity_scores.trigram_jaccard * 100
                        ).toFixed(2)}%</div>
                    </div>
                    <div class="similarity-item">
                        <div class="similarity-label">Cosine Similarity</div>
                        <div class="similarity-value">${(
                          item.result.similarity_scores.cosine_similarity * 100
                        ).toFixed(2)}%</div>
                    </div>
                </div>
            </div>
        `;
  });

  showResults(content);
}

function getScoreClass(probability) {
  if (probability >= 0.7) return "score-high";
  if (probability >= 0.4) return "score-medium";
  return "score-low";
}
