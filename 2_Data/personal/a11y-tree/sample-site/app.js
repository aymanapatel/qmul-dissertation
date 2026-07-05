("use strict");

// ---------- Accordion ----------
const triggers = document.querySelectorAll(".acc-trigger");
triggers.forEach((btn) => {
  btn.addEventListener("click", () => {
    const expanded = btn.getAttribute("aria-expanded") === "true";
    btn.setAttribute("aria-expanded", String(!expanded));
    const panel = document.getElementById(btn.getAttribute("aria-controls"));
    if (panel) panel.hidden = expanded;
  });
});

// ---------- Theme toggle ----------
const themeBtn = document.querySelector(".theme-toggle");
const saved = localStorage.getItem("theme");
if (saved) document.documentElement.setAttribute("data-theme", saved);

themeBtn.addEventListener("click", () => {
  const pressed = themeBtn.getAttribute("aria-pressed") === "true";
  const next = !pressed;
  themeBtn.setAttribute("aria-pressed", String(next));
  document.documentElement.setAttribute("data-theme", next ? "dark" : "light");
  themeBtn.textContent = next ? "Light theme" : "Dark theme";
  themeBtn.setAttribute("aria-label", next ? "Switch to light theme" : "Switch to dark theme");
  localStorage.setItem("theme", next ? "dark" : "light");
});
if (document.documentElement.getAttribute("data-theme") === "dark") {
  themeBtn.setAttribute("aria-pressed", "true");
  themeBtn.textContent = "Light theme";
  themeBtn.setAttribute("aria-label", "Switch to light theme");
}

// ---------- Dialog ----------
const dialog = document.getElementById("dir-dialog");
const openBtn = document.getElementById("open-dialog-btn");
openBtn.addEventListener("click", () => {
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
});
dialog.addEventListener("click", (e) => {
  if (e.target === dialog) dialog.close();
});

// ---------- Form validation ----------
const form = document.getElementById("signup-form");
const status = document.getElementById("form-status");
const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

form.addEventListener("submit", (e) => {
  e.preventDefault();
  let ok = true;

  const name = document.getElementById("name");
  const nameErr = document.getElementById("name-error");
  if (!name.value.trim()) {
    nameErr.hidden = false;
    name.setAttribute("aria-invalid", "true");
    ok = false;
  } else {
    nameErr.hidden = true;
    name.removeAttribute("aria-invalid");
  }

  const email = document.getElementById("email");
  const emailErr = document.getElementById("email-error");
  if (!emailRe.test(email.value.trim())) {
    emailErr.hidden = false;
    email.setAttribute("aria-invalid", "true");
    ok = false;
  } else {
    emailErr.hidden = true;
    email.removeAttribute("aria-invalid");
  }

  const agree = document.getElementById("agree");
  if (!agree.checked) {
    ok = false;
    agree.setAttribute("aria-invalid", "true");
  } else {
    agree.removeAttribute("aria-invalid");
  }

  if (ok) {
    status.hidden = false;
    status.textContent = "Thanks! Check your inbox to confirm.";
    form.reset();
  } else {
    status.hidden = false;
    status.textContent = "Please fix the highlighted fields.";
  }
});

[name, email].forEach((input) => {
  input.addEventListener("input", () => input.removeAttribute("aria-invalid"));
});