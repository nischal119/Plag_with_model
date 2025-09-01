/**
 * Authentication JavaScript
 * Handles form validation, password visibility, and interactive features
 */

// Wait for DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
  initializeAuthFeatures();
});

function initializeAuthFeatures() {
  // Initialize password toggle functionality
  initializePasswordToggles();

  // Initialize form validation
  initializeFormValidation();

  // Initialize flash message auto-dismiss
  initializeFlashMessages();

  // Initialize user menu dropdown
  initializeUserMenu();

  // Initialize real-time validation
  initializeRealTimeValidation();
}

/**
 * Password Toggle Functionality
 */
function initializePasswordToggles() {
  const passwordToggles = document.querySelectorAll(".password-toggle");
  passwordToggles.forEach((toggle) => {
    toggle.addEventListener("click", function (e) {
      e.preventDefault();
      const input = this.previousElementSibling;
      const icon = this.querySelector("i");

      if (input.type === "password") {
        input.type = "text";
        icon.className = "fas fa-eye-slash";
      } else {
        input.type = "password";
        icon.className = "fas fa-eye";
      }
    });
  });
}

function togglePassword(inputId) {
  const input = document.getElementById(inputId);
  const eyeIcon = document.getElementById(inputId + "-eye");

  if (input.type === "password") {
    input.type = "text";
    eyeIcon.className = "fas fa-eye-slash";
  } else {
    input.type = "password";
    eyeIcon.className = "fas fa-eye";
  }
}

/**
 * Form Validation
 */
function initializeFormValidation() {
  const forms = document.querySelectorAll(".auth-form");
  forms.forEach((form) => {
    form.addEventListener("submit", function (e) {
      if (!validateForm(this)) {
        e.preventDefault();
        return false;
      }
    });
  });
}

function validateForm(form) {
  let isValid = true;
  const inputs = form.querySelectorAll("input[required]");

  inputs.forEach((input) => {
    if (!validateInput(input)) {
      isValid = false;
    }
  });

  // Validate password confirmation if present
  const password = form.querySelector('input[name="password"]');
  const password2 = form.querySelector('input[name="password2"]');

  if (password && password2) {
    if (password.value !== password2.value) {
      showInputError(password2, "Passwords do not match");
      isValid = false;
    } else {
      clearInputError(password2);
    }
  }

  return isValid;
}

function validateInput(input) {
  const value = input.value.trim();
  const type = input.type;
  const name = input.name;

  // Clear previous errors
  clearInputError(input);

  // Required validation
  if (input.hasAttribute("required") && !value) {
    showInputError(input, "This field is required");
    return false;
  }

  // Email validation
  if (type === "email" && value) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(value)) {
      showInputError(input, "Please enter a valid email address");
      return false;
    }
  }

  // Password validation
  if (name === "password" && value) {
    if (value.length < 6) {
      showInputError(input, "Password must be at least 6 characters long");
      return false;
    }
  }

  // Username validation
  if (name === "username" && value) {
    if (value.length < 3) {
      showInputError(input, "Username must be at least 3 characters long");
      return false;
    }
    if (!/^[a-zA-Z0-9_]+$/.test(value)) {
      showInputError(
        input,
        "Username can only contain letters, numbers, and underscores"
      );
      return false;
    }
  }

  return true;
}

function showInputError(input, message) {
  input.classList.add("error");

  // Remove existing error message
  const existingError = input.parentNode.querySelector(".error-message");
  if (existingError) {
    existingError.remove();
  }

  // Create new error message
  const errorElement = document.createElement("span");
  errorElement.className = "error-message";
  errorElement.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;

  // Insert after the input (or password container)
  const container = input.closest(".password-input-container") || input;
  container.parentNode.insertBefore(errorElement, container.nextSibling);
}

function clearInputError(input) {
  input.classList.remove("error");
  const errorMessage = input.parentNode.querySelector(".error-message");
  if (errorMessage) {
    errorMessage.remove();
  }
}

/**
 * Real-time Validation
 */
function initializeRealTimeValidation() {
  const inputs = document.querySelectorAll("input");
  inputs.forEach((input) => {
    input.addEventListener("blur", function () {
      if (this.value.trim()) {
        validateInput(this);
      }
    });

    input.addEventListener("input", function () {
      // Clear error on input if it exists
      if (this.classList.contains("error")) {
        clearInputError(this);
      }
    });
  });
}

/**
 * Flash Messages
 */
function initializeFlashMessages() {
  const flashMessages = document.querySelectorAll(".flash-message");

  // Auto-dismiss after 5 seconds
  flashMessages.forEach((message) => {
    setTimeout(() => {
      if (message.parentNode) {
        message.style.animation = "slideOut 0.3s ease-in";
        setTimeout(() => {
          if (message.parentNode) {
            message.remove();
          }
        }, 300);
      }
    }, 5000);
  });

  // Add close button functionality
  const closeButtons = document.querySelectorAll(".flash-close");
  closeButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const message = this.closest(".flash-message");
      message.style.animation = "slideOut 0.3s ease-in";
      setTimeout(() => {
        if (message.parentNode) {
          message.remove();
        }
      }, 300);
    });
  });
}

/**
 * User Menu Dropdown
 */
function initializeUserMenu() {
  const userMenu = document.querySelector(".user-menu");
  if (!userMenu) return;

  const userInfo = userMenu.querySelector(".user-info");
  const dropdown = userMenu.querySelector(".nav-dropdown");

  if (!userInfo || !dropdown) return;

  let isOpen = false;
  let timeoutId = null;

  function showDropdown() {
    clearTimeout(timeoutId);
    dropdown.style.opacity = "1";
    dropdown.style.visibility = "visible";
    dropdown.style.transform = "translateY(0)";
    isOpen = true;
  }

  function hideDropdown() {
    timeoutId = setTimeout(() => {
      dropdown.style.opacity = "0";
      dropdown.style.visibility = "hidden";
      dropdown.style.transform = "translateY(-10px)";
      isOpen = false;
    }, 100);
  }

  userInfo.addEventListener("mouseenter", showDropdown);
  userMenu.addEventListener("mouseleave", hideDropdown);
  dropdown.addEventListener("mouseenter", () => clearTimeout(timeoutId));

  // Keyboard navigation
  userInfo.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (isOpen) {
        hideDropdown();
      } else {
        showDropdown();
      }
    }
  });

  // Close on click outside
  document.addEventListener("click", function (e) {
    if (!userMenu.contains(e.target) && isOpen) {
      hideDropdown();
    }
  });
}

/**
 * Password Strength Indicator
 */
function updatePasswordStrength(password, strengthElementId) {
  const strengthDiv = document.getElementById(strengthElementId);
  if (!strengthDiv) return;

  let strength = 0;
  let feedback = [];

  // Length check
  if (password.length >= 8) strength++;
  else if (password.length >= 6) strength += 0.5;
  else feedback.push("At least 6 characters");

  // Character type checks
  if (/[a-z]/.test(password)) strength++;
  else feedback.push("Lowercase letter");

  if (/[A-Z]/.test(password)) strength++;
  else feedback.push("Uppercase letter");

  if (/[0-9]/.test(password)) strength++;
  else feedback.push("Number");

  if (/[^a-zA-Z0-9]/.test(password)) strength++;
  else feedback.push("Special character");

  // Common patterns (reduce strength)
  if (/(.)\1{2,}/.test(password)) strength -= 0.5; // Repeated characters
  if (/123|abc|qwe/i.test(password)) strength -= 0.5; // Common sequences

  strength = Math.max(0, Math.min(5, strength));

  const levels = ["Very Weak", "Weak", "Fair", "Good", "Strong"];
  const colors = ["#ff4757", "#ff6b35", "#ffa502", "#2ed573", "#5f27cd"];

  if (password.length === 0) {
    strengthDiv.innerHTML = "";
  } else {
    const strengthIndex = Math.floor(strength);
    const strengthColor = colors[strengthIndex] || colors[0];
    const strengthText = levels[strengthIndex] || levels[0];
    const strengthWidth = (strength / 5) * 100;

    strengthDiv.innerHTML = `
            <div class="strength-bar">
                <div class="strength-fill" style="width: ${strengthWidth}%; background-color: ${strengthColor}"></div>
            </div>
            <span class="strength-text" style="color: ${strengthColor}">${strengthText}</span>
            ${
              feedback.length > 0
                ? `<div class="strength-feedback">Missing: ${feedback.join(
                    ", "
                  )}</div>`
                : ""
            }
        `;
  }
}

/**
 * Form Utilities
 */
function showFormLoading(form) {
  const submitButton = form.querySelector(
    'input[type="submit"], button[type="submit"]'
  );
  if (submitButton) {
    submitButton.disabled = true;
    submitButton.innerHTML =
      '<i class="fas fa-spinner fa-spin"></i> Processing...';
  }
}

function hideFormLoading(form) {
  const submitButton = form.querySelector(
    'input[type="submit"], button[type="submit"]'
  );
  if (submitButton) {
    submitButton.disabled = false;
    submitButton.innerHTML = submitButton.dataset.originalText || "Submit";
  }
}

/**
 * AJAX Form Submission (for future use)
 */
function submitFormAsync(form, callback) {
  const formData = new FormData(form);
  const url = form.action || window.location.href;

  showFormLoading(form);

  fetch(url, {
    method: "POST",
    body: formData,
    headers: {
      "X-Requested-With": "XMLHttpRequest",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      hideFormLoading(form);
      if (callback) {
        callback(data);
      }
    })
    .catch((error) => {
      hideFormLoading(form);
      console.error("Form submission error:", error);
      showNotification("An error occurred. Please try again.", "error");
    });
}

/**
 * Notification System
 */
function showNotification(message, type = "info") {
  const notificationHtml = `
        <div class="flash-message flash-${type}" style="animation: slideIn 0.3s ease-out;">
            <i class="fas fa-${
              type === "error"
                ? "exclamation-triangle"
                : type === "success"
                ? "check-circle"
                : "info-circle"
            }"></i>
            ${message}
            <button class="flash-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

  let flashContainer = document.querySelector(".flash-messages");
  if (!flashContainer) {
    flashContainer = document.createElement("div");
    flashContainer.className = "flash-messages";
    document
      .querySelector(".container")
      .insertBefore(
        flashContainer,
        document.querySelector(".container").firstChild
      );
  }

  flashContainer.insertAdjacentHTML("beforeend", notificationHtml);

  // Auto-dismiss after 5 seconds
  const newMessage = flashContainer.lastElementChild;
  setTimeout(() => {
    if (newMessage.parentNode) {
      newMessage.remove();
    }
  }, 5000);
}

/**
 * Mobile Menu Toggle (for responsive design)
 */
function toggleMobileMenu() {
  const nav = document.querySelector(".user-nav");
  nav.classList.toggle("mobile-open");
}

/**
 * Accessibility Improvements
 */
function initializeAccessibility() {
  // Add ARIA labels to interactive elements
  const passwordToggles = document.querySelectorAll(".password-toggle");
  passwordToggles.forEach((toggle) => {
    toggle.setAttribute("aria-label", "Toggle password visibility");
  });

  // Add focus indicators
  const focusableElements = document.querySelectorAll("input, button, a");
  focusableElements.forEach((element) => {
    element.addEventListener("focus", function () {
      this.style.outline = "2px solid #667eea";
      this.style.outlineOffset = "2px";
    });

    element.addEventListener("blur", function () {
      this.style.outline = "";
      this.style.outlineOffset = "";
    });
  });
}

/**
 * Animation utilities
 */
const animationStyles = `
    @keyframes slideOut {
        from {
            opacity: 1;
            transform: translateY(0);
        }
        to {
            opacity: 0;
            transform: translateY(-20px);
        }
    }
    
    .strength-feedback {
        font-size: 0.75rem;
        color: #666;
        margin-top: 0.25rem;
    }
`;

// Inject animation styles
const styleSheet = document.createElement("style");
styleSheet.textContent = animationStyles;
document.head.appendChild(styleSheet);

// Initialize accessibility features
document.addEventListener("DOMContentLoaded", initializeAccessibility);
