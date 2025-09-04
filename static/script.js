// Global variables
let currentMode = "pairwise";
let referenceFileCounter = 0;
let referenceFiles = new Map(); // Map to store reference file data

// DOM elements
const modeBtns = document.querySelectorAll(".mode-btn");
const modeContents = document.querySelectorAll(".mode-content");
const trainBtn = document.getElementById("trainBtn");
const resetBtn = document.getElementById("resetBtn");
const compareBtn = document.getElementById("compareBtn");
const multipleBtn = document.getElementById("multipleBtn");
const addReferenceBtn = document.getElementById("addReferenceBtn");
const clearAllReferencesBtn = document.getElementById("clearAllReferencesBtn");
const bulkReferenceFiles = document.getElementById("bulkReferenceFiles");
const referenceFilesContainer = document.getElementById(
  "referenceFilesContainer"
);
const trainingStatus = document.getElementById("trainingStatus");
const modelStatusText = document.getElementById("modelStatusText");
const modelStatusIcon = document.getElementById("modelStatusIcon");
const resultsSection = document.getElementById("resultsSection");
const resultsContent = document.getElementById("resultsContent");
const loadingOverlay = document.getElementById("loadingOverlay");

// File input elements
const file1 = document.getElementById("file1");
const file2 = document.getElementById("file2");
const multipleFile = document.getElementById("multipleFile");

// File name display elements
const file1Name = document.getElementById("file1Name");
const file2Name = document.getElementById("file2Name");
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
  multipleBtn.addEventListener("click", compareMultiple);

  // Reference file management
  addReferenceBtn.addEventListener("click", addReferenceFile);
  clearAllReferencesBtn.addEventListener("click", clearAllReferenceFiles);
  bulkReferenceFiles.addEventListener("change", handleBulkReferenceUpload);
}

function setupFileUploads() {
  // File upload handlers
  file1.addEventListener("change", (e) => handleFileUpload(e, file1Name));
  file2.addEventListener("change", (e) => handleFileUpload(e, file2Name));
  multipleFile.addEventListener("change", (e) =>
    handleFileUpload(e, multipleFileName)
  );

  // Add event listeners to textareas to clear file inputs when manually typing
  const text1 = document.getElementById("text1");
  const text2 = document.getElementById("text2");
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

  // Clear reference files without confirmation
  clearAllReferenceFiles(false);

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

function addReferenceFile() {
  referenceFileCounter++;
  const fileId = `referenceFile${referenceFileCounter}`;

  const fileItem = document.createElement("div");
  fileItem.className = "reference-file-item";
  fileItem.id = `referenceItem${referenceFileCounter}`;

  fileItem.innerHTML = `
    <div class="file-info">
      <i class="fas fa-file-alt file-icon"></i>
      <span class="file-name">No file selected</span>
      <span class="file-size"></span>
    </div>
    <div class="file-upload">
      <input type="file" id="${fileId}" accept=".txt,.docx,.pdf" style="display: none;">
      <button class="btn btn-secondary" onclick="document.getElementById('${fileId}').click()">
        <i class="fas fa-upload"></i> Upload
      </button>
      <span id="${fileId}Name"></span>
    </div>
    <button class="remove-btn" onclick="removeReferenceFile(${referenceFileCounter})">
      <i class="fas fa-times"></i>
    </button>
  `;

  referenceFilesContainer.appendChild(fileItem);

  // Add event listener for the new file input
  const fileInput = document.getElementById(fileId);
  const fileNameSpan = document.getElementById(`${fileId}Name`);

  fileInput.addEventListener("change", (e) =>
    handleReferenceFileUpload(e, fileId, fileNameSpan)
  );

  // Store reference file data
  referenceFiles.set(fileId, {
    element: fileItem,
    fileInput: fileInput,
    fileNameSpan: fileNameSpan,
    file: null,
  });
}

function removeReferenceFile(counter) {
  const fileId = `referenceFile${counter}`;
  const fileData = referenceFiles.get(fileId);

  if (fileData) {
    fileData.element.remove();
    referenceFiles.delete(fileId);
  }
}

function handleReferenceFileUpload(event, fileId, nameElement) {
  const file = event.target.files[0];
  const fileData = referenceFiles.get(fileId);

  if (file && fileData) {
    // Update file data
    fileData.file = file;
    fileData.fileNameSpan.textContent = file.name;
    fileData.fileNameSpan.style.color = "#38a169";

    // Update file info display
    const fileInfo = fileData.element.querySelector(".file-info");
    const fileName = fileInfo.querySelector(".file-name");
    const fileSize = fileInfo.querySelector(".file-size");

    fileName.textContent = file.name;
    fileName.style.color = "#2d3748";
    fileSize.textContent = formatFileSize(file.size);

    // Update button text
    updateCompareButtonText();
  } else if (fileData) {
    // Clear file data
    fileData.file = null;
    fileData.fileNameSpan.textContent = "";

    const fileInfo = fileData.element.querySelector(".file-info");
    const fileName = fileInfo.querySelector(".file-name");
    const fileSize = fileInfo.querySelector(".file-size");

    fileName.textContent = "No file selected";
    fileName.style.color = "#718096";
    fileSize.textContent = "";

    updateCompareButtonText();
  }
}

function handleBulkReferenceUpload(event) {
  const files = event.target.files;

  if (files.length === 0) return;

  // Clear existing reference files if any
  if (referenceFiles.size > 0) {
    if (!confirm("This will replace all existing reference files. Continue?")) {
      event.target.value = "";
      return;
    }
    clearAllReferenceFiles();
  }

  Array.from(files).forEach((file, index) => {
    referenceFileCounter++;
    const fileId = `bulkReferenceFile${referenceFileCounter}`;

    const fileItem = document.createElement("div");
    fileItem.className = "reference-file-item";
    fileItem.id = `referenceItem${referenceFileCounter}`;

    fileItem.innerHTML = `
      <div class="file-info">
        <i class="fas fa-file-alt file-icon"></i>
        <span class="file-name">${file.name}</span>
        <span class="file-size">${formatFileSize(file.size)}</span>
      </div>
      <div class="file-upload">
        <input type="file" id="${fileId}" accept=".txt,.docx,.pdf" style="display: none;">
        <button class="btn btn-secondary" onclick="document.getElementById('${fileId}').click()">
          <i class="fas fa-upload"></i> Change
        </button>
        <span id="${fileId}Name"></span>
      </div>
      <button class="remove-btn" onclick="removeReferenceFile(${referenceFileCounter})">
        <i class="fas fa-times"></i>
      </button>
    `;

    referenceFilesContainer.appendChild(fileItem);

    // Add event listener for the new file input
    const fileInput = document.getElementById(fileId);
    const fileNameSpan = document.getElementById(`${fileId}Name`);

    fileInput.addEventListener("change", (e) =>
      handleReferenceFileUpload(e, fileId, fileNameSpan)
    );

    // Store reference file data with the actual file
    referenceFiles.set(fileId, {
      element: fileItem,
      fileInput: fileInput,
      fileNameSpan: fileNameSpan,
      file: file,
    });

    // Update file info display
    const fileInfo = fileItem.querySelector(".file-info");
    const fileName = fileInfo.querySelector(".file-name");
    const fileSize = fileInfo.querySelector(".file-size");

    fileName.textContent = file.name;
    fileName.style.color = "#2d3748";
    fileSize.textContent = formatFileSize(file.size);
  });

  // Clear the input for future use
  event.target.value = "";

  updateCompareButtonText();
}

function clearAllReferenceFiles(requireConfirmation = true) {
  if (requireConfirmation && referenceFiles.size > 0) {
    if (
      !confirm(
        "Are you sure you want to clear all reference files? This action cannot be undone."
      )
    ) {
      return;
    }
  }

  referenceFiles.forEach((fileData) => {
    fileData.element.remove();
  });
  referenceFiles.clear();
  referenceFileCounter = 0;
  updateCompareButtonText();
}

function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
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

  // Classification mode removed

  // Check multiple reference mode
  const multipleFile = document.getElementById("multipleFile");
  const multipleBtn = document.getElementById("multipleBtn");

  // Count reference files with actual files
  let referenceFileCount = 0;
  referenceFiles.forEach((fileData) => {
    if (fileData.file) {
      referenceFileCount++;
    }
  });

  if (multipleFile.files.length > 0 || referenceFileCount > 0) {
    const fileText = multipleFile.files.length > 0 ? "Document" : "Text";
    const referenceText =
      referenceFileCount > 0
        ? ` (${referenceFileCount} reference${
            referenceFileCount > 1 ? "s" : ""
          })`
        : "";
    multipleBtn.innerHTML = `<i class="fas fa-file-alt"></i> Compare ${fileText} Against References${referenceText}`;
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

    // Add original text/file
    if (!text && !multipleFile.files[0]) {
      alert("Please provide original text or file for comparison");
      hideLoading();
      return;
    }

    formData.append("text", text);
    if (multipleFile.files[0]) {
      formData.append("file", multipleFile.files[0]);
    }

    // Add reference files
    let referenceFileCount = 0;
    referenceFiles.forEach((fileData, fileId) => {
      if (fileData.file) {
        formData.append(`reference_file_${referenceFileCount}`, fileData.file);
        formData.append(
          `reference_name_${referenceFileCount}`,
          fileData.file.name
        );
        referenceFileCount++;
      }
    });

    if (referenceFileCount === 0) {
      alert("Please upload at least one reference document");
      hideLoading();
      return;
    }

    formData.append("reference_count", referenceFileCount);

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
                      data.result.overall_similarity
                    )}">
                        ${(data.result.overall_similarity * 100).toFixed(1)}%
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
                            <span class="summary-label">Overall Similarity:</span>
                            <span class="summary-value">${(
                              data.result.overall_similarity * 100
                            ).toFixed(1)}%</span>
                        </div>
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
                                                          match.highlighted_context1 ||
                                                          match.context1 ||
                                                          "No context available"
                                                        }</div>
                                                    </div>
                                                    <div class="context-item">
                                                        <strong>Text 2 Context:</strong>
                                                        <div class="context-text">${
                                                          match.highlighted_context2 ||
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
  }

  showResults(content);
}

function displayMultipleResults(data) {
  let content = `
        <div class="result-card">
            <h3><i class="fas fa-file-alt"></i> Multiple Reference Comparison Results</h3>
            <div class="original-document-info">
                <h4><i class="fas fa-file-alt"></i> Original Document</h4>
                <p><strong>Name:</strong> ${
                  data.original_file_name || "Text Input"
                }</p>
                <p><strong>Content Preview:</strong> ${data.input_text.substring(
                  0,
                  150
                )}${data.input_text.length > 150 ? "..." : ""}</p>
            </div>
        </div>
    `;

  data.results.forEach((item, index) => {
    // Create compact summary
    const summaryText = createCompactSummary(item);

    content += `
            <div class="result-card">
                <div class="result-header">
                    <div class="result-title">
                        <i class="fas fa-file-alt"></i> ${item.reference_name}
                    </div>
                    <div class="result-score ${getScoreClass(
                      item.result.plagiarism_probability
                    )}">
                        ${(item.result.plagiarism_probability * 100).toFixed(
                          1
                        )}%
                    </div>
                </div>
                
                <div class="compact-summary">
                    <div class="summary-badge ${
                      item.result.is_plagiarized ? "plagiarized" : "original"
                    }">
                        <i class="fas ${
                          item.result.is_plagiarized
                            ? "fa-exclamation-triangle"
                            : "fa-check-circle"
                        }"></i>
                        ${
                          item.result.is_plagiarized
                            ? "Likely Plagiarized"
                            : "Likely Original"
                        }
                    </div>
                    <div class="summary-text">${summaryText}</div>
                </div>

                <div class="results-tabs">
                    <div class="tab-buttons">
                        <button class="tab-btn active" onclick="switchTab(${index}, 'original')">
                            <i class="fas fa-file-alt"></i> Original Text
                        </button>
                        <button class="tab-btn" onclick="switchTab(${index}, 'matches')">
                            <i class="fas fa-search"></i> Matching Phrases (${
                              item.matches.length
                            })
                        </button>
                        <button class="tab-btn" onclick="switchTab(${index}, 'reference')">
                            <i class="fas fa-file-alt"></i> ${
                              item.reference_name
                            }
                        </button>
                    </div>
                    
                    <div class="tab-content">
                        <div id="tab-${index}-original" class="tab-pane active">
                            <div class="highlighted-text">
                                <h4><i class="fas fa-highlighter"></i> Original Document with Highlights</h4>
                                <div class="text-content">${
                                  item.highlighted_text1
                                }</div>
                            </div>
                        </div>
                        
                        <div id="tab-${index}-matches" class="tab-pane">
                            <div class="matching-phrases-section">
                                <h4><i class="fas fa-search"></i> Matching Phrases Found</h4>
                                <div class="matching-phrases-grid">
                                    ${(() => {
                                      const exactMatches = item.matches.filter(
                                        (m) => m.match_type === "exact"
                                      );
                                      const semanticMatches =
                                        item.matches.filter(
                                          (m) => m.match_type === "semantic"
                                        );

                                      let content = "";

                                      if (exactMatches.length > 0) {
                                        content +=
                                          '<div class="match-group"><h5><i class="fas fa-check-circle"></i> Exact Matches (${exactMatches.length})</h5>';
                                        exactMatches
                                          .slice(0, 5)
                                          .forEach((match) => {
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
                                                                    <strong>Original Context:</strong>
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

                                      if (semanticMatches.length > 0) {
                                        content +=
                                          '<div class="match-group"><h5><i class="fas fa-sync-alt"></i> Semantic Matches (${semanticMatches.length})</h5>';
                                        semanticMatches
                                          .slice(0, 5)
                                          .forEach((match) => {
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
                                                                <strong>Original:</strong> "${
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
                                                                  match.similarity *
                                                                  100
                                                                ).toFixed(1)}%
                                                            </div>
                                                        </div>
                                                    </div>
                                                `;
                                          });
                                        content += "</div>";
                                      }

                                      if (
                                        exactMatches.length === 0 &&
                                        semanticMatches.length === 0
                                      ) {
                                        content =
                                          '<div class="no-matches"><i class="fas fa-info-circle"></i> No matching phrases found</div>';
                                      }

                                      return content;
                                    })()}
                                </div>
                            </div>
                        </div>
                        
                        <div id="tab-${index}-reference" class="tab-pane">
                            <div class="highlighted-text">
                                <h4><i class="fas fa-highlighter"></i> ${
                                  item.reference_name
                                } with Highlights</h4>
                                <div class="text-content">${
                                  item.highlighted_text2
                                }</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
  });

  showResults(content);
}

function createCompactSummary(item) {
  const exactCount = item.summary.exact_matches;
  const semanticCount = item.summary.semantic_matches;
  const totalCount = item.summary.total_matches;
  const longestMatch = item.summary.longest_match;

  let summary = "";

  if (totalCount === 0) {
    summary = "No matching content found between documents.";
  } else {
    summary = `Found ${totalCount} matching phrases: `;

    if (exactCount > 0) {
      summary += `${exactCount} exact match${exactCount > 1 ? "es" : ""}`;
    }

    if (exactCount > 0 && semanticCount > 0) {
      summary += " and ";
    }

    if (semanticCount > 0) {
      summary += `${semanticCount} semantic match${
        semanticCount > 1 ? "es" : ""
      }`;
    }

    if (longestMatch > 1) {
      summary += `. Longest match: ${longestMatch} words.`;
    }
  }

  return summary;
}

function switchTab(resultIndex, tabName) {
  // Remove active class from all tabs and panes
  const tabButtons = document.querySelectorAll(
    `.result-card:nth-child(${resultIndex + 2}) .tab-btn`
  );
  const tabPanes = document.querySelectorAll(
    `.result-card:nth-child(${resultIndex + 2}) .tab-pane`
  );

  tabButtons.forEach((btn) => btn.classList.remove("active"));
  tabPanes.forEach((pane) => pane.classList.remove("active"));

  // Add active class to selected tab and pane
  const selectedTab = document.querySelector(
    `.result-card:nth-child(${resultIndex + 2}) .tab-btn[onclick*="${tabName}"]`
  );
  const selectedPane = document.getElementById(`tab-${resultIndex}-${tabName}`);

  if (selectedTab) selectedTab.classList.add("active");
  if (selectedPane) selectedPane.classList.add("active");
}

function getScoreClass(probability) {
  if (probability >= 0.7) return "score-high";
  if (probability >= 0.4) return "score-medium";
  return "score-low";
}
