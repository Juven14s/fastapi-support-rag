"use strict";

const form = document.querySelector("#question-form");
const question = document.querySelector("#question");
const submit = document.querySelector("#submit-question");
const errorMessage = document.querySelector("#request-error");
const progress = document.querySelector("#progress");
const result = document.querySelector("#result");
const resultTitle = document.querySelector("#result-title");
const answer = document.querySelector("#answer");
const latency = document.querySelector("#latency");
const sourceSection = document.querySelector("#source-section");
const sourceList = document.querySelector("#sources");
const examples = Array.from(document.querySelectorAll(".example"));
const durationFormat = new Intl.NumberFormat("fr-FR", {maximumFractionDigits: 1});

function setBusy(busy) {
  submit.disabled = busy;
  question.readOnly = busy;
  examples.forEach((button) => { button.disabled = busy; });
  progress.hidden = !busy;
}

function sourceUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol === "https:") return url.href;
  } catch {
    // A malformed link still leaves the title and excerpt readable.
  }
  return null;
}

function presentSources(citations) {
  const byId = new Map();
  const numbered = [];
  sourceList.replaceChildren();
  citations.forEach((citation, index) => {
    const number = index + 1;
    const item = document.createElement("li");
    item.className = "source";
    item.id = `source-${number}`;
    const url = sourceUrl(citation.url);
    const title = document.createElement(url ? "a" : "span");
    title.className = "source-title";
    title.textContent = `${number}. ${citation.title}${url ? " ↗" : ""}`;
    if (url) {
      title.href = url;
      title.target = "_blank";
      title.rel = "noopener noreferrer";
    }
    item.append(title);
    if (citation.excerpt) {
      const excerpt = document.createElement("p");
      excerpt.className = "source-excerpt";
      const cleanText = citation.excerpt.replace(/\s+/gu, " ").trim();
      excerpt.textContent = cleanText.length > 260 ? cleanText.slice(0, 257) + "…" : cleanText;
      item.append(excerpt);
    }
    sourceList.append(item);
    const marker = {number, id: item.id};
    byId.set(String(citation.id), marker);
    numbered.push(marker);
  });
  sourceSection.hidden = numbered.length === 0;
  return {byId, numbered};
}

function appendInline(parent, value, markers) {
  for (const part of value.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]{1,180}\])/gu)) {
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      const code = document.createElement("code");
      code.textContent = part.slice(1, -1);
      parent.append(code);
    } else if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      parent.append(strong);
    } else {
      const label = part.slice(1, -1);
      const marker = part.startsWith("[") && part.endsWith("]") &&
        (markers.byId.get(label) || (/^[1-9]\d*$/u.test(label) && markers.numbered[Number(label) - 1]));
      if (marker) {
        const link = document.createElement("a");
        link.className = "citation";
        link.href = `#${marker.id}`;
        link.textContent = `[${marker.number}]`;
        link.setAttribute("aria-label", `Aller au passage ${marker.number}`);
        parent.append(link);
      } else {
        parent.append(document.createTextNode(part));
      }
    }
  }
}

function presentAnswer(value, markers) {
  answer.replaceChildren();
  const blocks = value.replace(/\r\n/g, "\n").split(/(```[^\n]*\n[\s\S]*?(?:```|$))/gu);
  for (const block of blocks) {
    if (block.startsWith("```") && block.includes("\n")) {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = block.slice(block.indexOf("\n") + 1).replace(/\n?```$/u, "");
      pre.append(code);
      answer.append(pre);
      continue;
    }
    for (const segment of block.trim().split(/\n\s*\n/u)) {
      if (!segment) continue;
      const paragraph = document.createElement("p");
      appendInline(paragraph, segment.replace(/^#{1,6}\s+/gmu, ""), markers);
      answer.append(paragraph);
    }
  }
}

function showAnswer(data) {
  if (!["answered", "abstained"].includes(data.status) || typeof data.answer !== "string") {
    throw new Error("Format de réponse inattendu.");
  }
  result.dataset.status = data.status;
  resultTitle.textContent = data.status === "answered" ? "Réponse" : "Informations insuffisantes";
  latency.textContent = Number.isFinite(data.latency_ms)
    ? `${durationFormat.format(data.latency_ms / 1000)} s`
    : "";
  const markers = presentSources(Array.isArray(data.citations) ? data.citations : []);
  presentAnswer(data.answer, markers);
  result.hidden = false;
  resultTitle.focus();
}

for (const example of examples) {
  example.addEventListener("click", () => {
    question.value = example.textContent.trim();
    question.setCustomValidity("");
    question.focus();
  });
}
question.addEventListener("input", () => { question.setCustomValidity(""); });
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submit.disabled) return;
  const text = question.value.trim();
  if (text.length < 3) {
    question.setCustomValidity("Saisis au moins trois caractères.");
    question.reportValidity();
    return;
  }

  errorMessage.hidden = true;
  result.hidden = true;
  setBusy(true);
  try {
    const response = await fetch("/v1/answer", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: text}),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    showAnswer(await response.json());
  } catch {
    errorMessage.textContent = "Le service ne répond pas pour le moment. Réessaie dans un instant.";
    errorMessage.hidden = false;
  } finally {
    setBusy(false);
  }
});
