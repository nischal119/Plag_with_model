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

  // Add event listeners to textareas to clear file inputs when manually typing
  const text1 = document.getElementById("text1");
  const text2 = document.getElementById("text2");
  const classifyText = document.getElementById("classifyText");
  const multipleText = document.getElementById("multipleText");

  text1.addEventListener("input", () => {
    if (text1.value && !text1.value.startsWith("📄")) {
      file1.value = "";
      file1Name.textContent = "";
      updateCompareButtonText();
    }
  });

  text2.addEventListener("input", () => {
    if (text2.value && !text2.value.startsWith("📄")) {
      file2.value = "";
      file2Name.textContent = "";
      updateCompareButtonText();
    }
  });

  classifyText.addEventListener("input", () => {
    if (classifyText.value && !classifyText.value.startsWith("📄")) {
      classifyFile.value = "";
      classifyFileName.textContent = "";
      updateCompareButtonText();
    }
  });

  multipleText.addEventListener("input", () => {
    if (multipleText.value && !multipleText.value.startsWith("📄")) {
      multipleFile.value = "";
      multipleFileName.textContent = "";
      updateCompareButtonText();
    }
  });
}

function clearAllInputs() {
  // Clear all textareas
  const textareas = document.querySelectorAll("textarea");
  textareas.forEach((textarea) => {
    textarea.value = "";
    textarea.style.fontStyle = "normal";
    textarea.style.color = "#2d3748";
  });

  // Clear all file inputs
  const fileInputs = document.querySelectorAll('input[type="file"]');
  fileInputs.forEach((input) => {
    input.value = "";
  });

  // Clear all file name displays
  const fileNameDisplays = document.querySelectorAll(".file-upload span");
  fileNameDisplays.forEach((span) => {
    span.textContent = "";
  });

  // Update button text
  updateCompareButtonText();
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

  // Clear all inputs when switching modes
  clearAllInputs();
}

function handleFileUpload(event, nameElement) {
  const file = event.target.files[0];
  if (file) {
    nameElement.textContent = file.name;
    nameElement.style.color = "#38a169";

    // Update the corresponding textarea with file name
    const textarea = event.target.parentElement.previousElementSibling;
    if (textarea && textarea.tagName === "TEXTAREA") {
      textarea.value = `📄 ${file.name}`;
      textarea.style.fontStyle = "italic";
      textarea.style.color = "#38a169";
    }

    // Update button text if it's a compare button
    updateCompareButtonText();
  } else {
    nameElement.textContent = "";

    // Clear the textarea if file is removed
    const textarea = event.target.parentElement.previousElementSibling;
    if (textarea && textarea.tagName === "TEXTAREA") {
      textarea.value = "";
      textarea.style.fontStyle = "normal";
      textarea.style.color = "#2d3748";
    }

    // Update button text
    updateCompareButtonText();
  }
}

function updateCompareButtonText() {
  // Check if any files are uploaded in pairwise mode
  const file1 = document.getElementById("file1");
  const file2 = document.getElementById("file2");
  const compareBtn = document.getElementById("compareBtn");

  if (file1.files.length > 0 || file2.files.length > 0) {
    compareBtn.innerHTML = '<i class="fas fa-file-alt"></i> Compare Documents';
  } else {
    compareBtn.innerHTML = '<i class="fas fa-search"></i> Compare Texts';
  }

  // Check classification mode
  const classifyFile = document.getElementById("classifyFile");
  const classifyBtn = document.getElementById("classifyBtn");

  if (classifyFile.files.length > 0) {
    classifyBtn.innerHTML = '<i class="fas fa-file-alt"></i> Classify Document';
  } else {
    classifyBtn.innerHTML = '<i class="fas fa-tag"></i> Classify Text';
  }

  // Check multiple reference mode
  const multipleFile = document.getElementById("multipleFile");
  const multipleBtn = document.getElementById("multipleBtn");

  if (multipleFile.files.length > 0) {
    multipleBtn.innerHTML =
      '<i class="fas fa-file-alt"></i> Compare Document Against References';
  } else {
    multipleBtn.innerHTML =
      '<i class="fas fa-list"></i> Compare Against References';
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

                <div class="matching-summary">
                    <h4><i class="fas fa-chart-pie"></i> Matching Summary</h4>
                    <div class="summary-stats">
                        <div class="summary-item">
                            <span class="summary-label">Total Matches:</span>
                            <span class="summary-value">${
                              data.summary.total_matches
                            }</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-label">Exact Matches:</span>
                            <span class="summary-value exact">${
                              data.summary.exact_matches
                            }</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-label">Semantic Matches:</span>
                            <span class="summary-value semantic">${
                              data.summary.semantic_matches
                            }</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-label">Longest Match:</span>
                            <span class="summary-value">${
                              data.summary.longest_match
                            } words</span>
                        </div>
                    </div>
                </div>

                <div class="matching-phrases-section">
                    <h4><i class="fas fa-search"></i> Matching Phrases & Context</h4>
                    <div class="matching-phrases-grid">
                        ${(() => {
                          // Group matches by type
                          const exactMatches = data.matches.filter(
                            (m) => m.match_type === "exact"
                          );
                          const semanticMatches = data.matches.filter(
                            (m) => m.match_type === "semantic"
                          );

                          let content = "";

                          // Show exact matches first
                          if (exactMatches.length > 0) {
                            content +=
                              '<div class="match-group"><h5><i class="fas fa-check-circle"></i> Exact Matches</h5>';
                            exactMatches.forEach((match) => {
                              content += `
                                        <div class="match-card exact">
                                            <div class="match-header">
                                                <div class="match-type exact">
                                                    <i class="fas fa-check-circle"></i>
                                                    Exact Match
                                                </div>
                                                <div class="match-length">${
                                                  match.length
                                                }-gram</div>
                                            </div>
                                            
                                            <div class="match-content">
                                                <div class="match-phrase">
                                                    <strong>Phrase:</strong> "${
                                                      match.phrase
                                                    }"
                                                </div>
                                                <div class="match-context">
                                                    <div class="context-item">
                                                        <strong>Text 1 Context:</strong>
                                                        <div class="context-text">${
                                                          match.context1 ||
                                                          "No context available"
                                                        }</div>
                                                    </div>
                                                    <div class="context-item">
                                                        <strong>Text 2 Context:</strong>
                                                        <div class="context-text">${
                                                          match.context2 ||
                                                          "No context available"
                                                        }</div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    `;
                            });
                            content += "</div>";
                          }

                          // Show semantic matches
                          if (semanticMatches.length > 0) {
                            content +=
                              '<div class="match-group"><h5><i class="fas fa-sync-alt"></i> Semantic Matches</h5>';
                            semanticMatches.forEach((match) => {
                              content += `
                                        <div class="match-card semantic">
                                            <div class="match-header">
                                                <div class="match-type semantic">
                                                    <i class="fas fa-sync-alt"></i>
                                                    Semantic Match
                                                </div>
                                                <div class="match-length">${
                                                  match.length
                                                }-gram</div>
                                            </div>
                                            
                                            <div class="match-content">
                                                <div class="match-phrase">
                                                    <strong>Text 1:</strong> "${
                                                      match.phrase1
                                                    }"
                                                </div>
                                                <div class="match-phrase">
                                                    <strong>Text 2:</strong> "${
                                                      match.phrase2
                                                    }"
                                                </div>
                                                <div class="match-similarity">
                                                    <strong>Similarity:</strong> ${(
                                                      match.similarity * 100
                                                    ).toFixed(1)}%
                                                </div>
                                            </div>
                                        </div>
                                    `;
                            });
                            content += "</div>";
                          }

                          return content;
                        })()}
                    </div>
                </div>

                <div class="highlighted-texts">
                    <div class="highlighted-text">
                        <h4><i class="fas fa-highlighter"></i> Text 1 (with highlights)</h4>
                        <div class="text-content">${
                          data.highlighted_text1
                        }</div>
                    </div>

                    <div class="highlighted-text">
                        <h4><i class="fas fa-highlighter"></i> Text 2 (with highlights)</h4>
                        <div class="text-content">${
                          data.highlighted_text2
                        }</div>
                    </div>
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

                <div class="matching-summary">
                    <h4><i class="fas fa-chart-pie"></i> Matching Summary</h4>
                    <div class="summary-stats">
                        <div class="summary-item">
                            <span class="summary-label">Total Matches:</span>
                            <span class="summary-value">${
                              item.summary.total_matches
                            }</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-label">Exact Matches:</span>
                            <span class="summary-value exact">${
                              item.summary.exact_matches
                            }</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-label">Semantic Matches:</span>
                            <span class="summary-value semantic">${
                              item.summary.semantic_matches
                            }</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-label">Longest Match:</span>
                            <span class="summary-value">${
                              item.summary.longest_match
                            } words</span>
                        </div>
                    </div>
                </div>

                ${
                  item.matches.length > 0
                    ? `
                <div class="matching-phrases-section">
                    <h4><i class="fas fa-search"></i> Top Matching Phrases</h4>
                    <div class="matching-phrases-grid">
                        ${(() => {
                          // Group matches by type
                          const exactMatches = item.matches.filter(
                            (m) => m.match_type === "exact"
                          );
                          const semanticMatches = item.matches.filter(
                            (m) => m.match_type === "semantic"
                          );

                          let content = "";

                          // Show exact matches first
                          if (exactMatches.length > 0) {
                            content +=
                              '<div class="match-group"><h5><i class="fas fa-check-circle"></i> Exact Matches</h5>';
                            exactMatches.slice(0, 3).forEach((match) => {
                              content += `
                                        <div class="match-card exact">
                                            <div class="match-header">
                                                <div class="match-type exact">
                                                    <i class="fas fa-check-circle"></i>
                                                    Exact Match
                                                </div>
                                                <div class="match-length">${
                                                  match.length
                                                }-gram</div>
                                            </div>
                                            
                                            <div class="match-content">
                                                <div class="match-phrase">
                                                    <strong>Phrase:</strong> "${
                                                      match.phrase
                                                    }"
                                                </div>
                                                <div class="match-context">
                                                    <div class="context-item">
                                                        <strong>Input Context:</strong>
                                                        <div class="context-text">${
                                                          match.context1 ||
                                                          "No context available"
                                                        }</div>
                                                    </div>
                                                    <div class="context-item">
                                                        <strong>Reference Context:</strong>
                                                        <div class="context-text">${
                                                          match.context2 ||
                                                          "No context available"
                                                        }</div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    `;
                            });
                            content += "</div>";
                          }

                          // Show semantic matches
                          if (semanticMatches.length > 0) {
                            content +=
                              '<div class="match-group"><h5><i class="fas fa-sync-alt"></i> Semantic Matches</h5>';
                            semanticMatches.slice(0, 3).forEach((match) => {
                              content += `
                                        <div class="match-card semantic">
                                            <div class="match-header">
                                                <div class="match-type semantic">
                                                    <i class="fas fa-sync-alt"></i>
                                                    Semantic Match
                                                </div>
                                                <div class="match-length">${
                                                  match.length
                                                }-gram</div>
                                            </div>
                                            
                                            <div class="match-content">
                                                <div class="match-phrase">
                                                    <strong>Input:</strong> "${
                                                      match.phrase1
                                                    }"
                                                </div>
                                                <div class="match-phrase">
                                                    <strong>Reference:</strong> "${
                                                      match.phrase2
                                                    }"
                                                </div>
                                                <div class="match-similarity">
                                                    <strong>Similarity:</strong> ${(
                                                      match.similarity * 100
                                                    ).toFixed(1)}%
                                                </div>
                                            </div>
                                        </div>
                                    `;
                            });
                            content += "</div>";
                          }

                          return content;
                        })()}
                    </div>
                </div>
                `
                    : ""
                }
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
