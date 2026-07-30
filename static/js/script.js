/**
 * script.js
 * ---------
 * Handles three things for CardioPredict:
 *   1. Dark/Light theme toggle (persisted in localStorage)
 *   2. Client-side form validation feedback (server still re-validates)
 *   3. Small UX animations: staggered reveals, animated risk number count-up
 */

(function initTheme() {
  const root = document.documentElement;
  const saved = localStorage.getItem("cardiopredict-theme");
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  const initial = saved || (prefersLight ? "light" : "dark");
  root.setAttribute("data-theme", initial);

  document.addEventListener("DOMContentLoaded", () => {
    const toggleBtn = document.getElementById("theme-toggle");
    if (!toggleBtn) return;

    toggleBtn.addEventListener("click", () => {
      const current = root.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("cardiopredict-theme", next);
    });
  });
})();

document.addEventListener("DOMContentLoaded", () => {
  // Stagger the .reveal animations slightly so sections don't all pop at once
  document.querySelectorAll(".reveal").forEach((el, i) => {
    if (!el.style.getPropertyValue("--delay")) {
      el.style.setProperty("--delay", `${i * 0.08}s`);
    }
  });

  setupFormValidation();
  animateGaugeNumber();
});

/**
 * Client-side validation mirrors the ranges enforced in predict.py's
 * FEATURE_SPECS. This gives instant feedback, but the server is the
 * source of truth - never trust the client alone.
 */
function setupFormValidation() {
  const form = document.getElementById("predict-form");
  if (!form) return;

  form.addEventListener("submit", (event) => {
    let hasError = false;

    form.querySelectorAll("input[required], select[required]").forEach((input) => {
      const field = input.closest(".field");
      const value = input.value.trim();
      let fieldValid = true;

      if (value === "") {
        fieldValid = false;
      } else if (input.tagName === "INPUT" && input.type === "number") {
        const num = parseFloat(value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        if (Number.isNaN(num) || num < min || num > max) {
          fieldValid = false;
        }
      }

      if (field) field.classList.toggle("field-invalid", !fieldValid);
      if (!fieldValid) hasError = true;
    });

    if (hasError) {
      event.preventDefault();
      const firstInvalid = form.querySelector(".field-invalid");
      if (firstInvalid) {
        firstInvalid.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  });

  // Clear the invalid state as soon as the user fixes a field
  form.querySelectorAll("input, select").forEach((input) => {
    input.addEventListener("input", () => {
      input.closest(".field")?.classList.remove("field-invalid");
    });
  });
}

/**
 * Animates the big risk percentage number counting up from 0 on the
 * result page, in sync with the SVG gauge fill animation in CSS.
 */
function animateGaugeNumber() {
  const el = document.querySelector(".gauge-number");
  if (!el) return;

  const target = parseFloat(el.dataset.target || "0");
  const duration = 1200;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    el.textContent = (target * eased).toFixed(1);
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = target.toFixed(1);
  }

  requestAnimationFrame(tick);
}
