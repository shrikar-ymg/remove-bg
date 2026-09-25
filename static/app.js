"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const elements = {
  addImagesButton: $("#addImagesButton"),
  backgroundOptions: $("#backgroundOptions"),
  browseButton: $("#browseButton"),
  brushCursor: $("#brushCursor"),
  brushSize: $("#brushSize"),
  brushValue: $("#brushValue"),
  canvasShell: $("#canvasShell"),
  canvasStack: $("#canvasStack"),
  canvasViewport: $("#canvasViewport"),
  clearProjectButton: $("#clearProjectButton"),
  compareDivider: $("#compareDivider"),
  compareSlider: $("#compareSlider"),
  customColor: $("#customColor"),
  downloadButton: $("#downloadButton"),
  downloadFormat: $("#downloadFormat"),
  dropStage: $("#dropStage"),
  editorStage: $("#editorStage"),
  emptyQueue: $("#emptyQueue"),
  feedbackControls: $("#feedbackControls"),
  fileInput: $("#fileInput"),
  fitButton: $("#fitButton"),
  historyLabel: $("#historyLabel"),
  loadingDetail: $("#loadingDetail"),
  loadingFile: $("#loadingFile"),
  loadingStage: $("#loadingStage"),
  loadingStep: $("#loadingStep"),
  loadingTitle: $("#loadingTitle"),
  newButton: $("#newButton"),
  originalCanvas: $("#originalCanvas"),
  progressBar: $("#progressBar"),
  preserveShadows: $("#preserveShadows"),
  qualityHint: $("#qualityHint"),
  qualityMode: $("#qualityMode"),
  queueCount: $("#queueCount"),
  queueList: $("#queueList"),
  redoButton: $("#redoButton"),
  removeButton: $("#removeButton"),
  resultControls: $("#resultControls"),
  resultMeta: $("#resultMeta"),
  resultName: $("#resultName"),
  resultStatus: $("#resultStatus"),
  runtimeBackend: $("#runtimeBackend"),
  saveFeedbackButton: $("#saveFeedbackButton"),
  selectionCanvas: $("#selectionCanvas"),
  shortcutsButton: $("#shortcutsButton"),
  shortcutsDialog: $("#shortcutsDialog"),
  startControls: $("#startControls"),
  startStatus: $("#startStatus"),
  toolMode: $("#toolMode"),
  undoButton: $("#undoButton"),
  zoomInButton: $("#zoomInButton"),
  zoomOutButton: $("#zoomOutButton"),
  zoomValue: $("#zoomValue"),
};

const canvas = $("#editorCanvas");
const context = canvas.getContext("2d", { willReadFrequently: false });
const originalContext = elements.originalCanvas.getContext("2d");
const selectionContext = elements.selectionCanvas.getContext("2d");
const HISTORY_LIMIT = 12;
const MAX_FILE_BYTES = 60 * 1024 * 1024;

const qualityCopy = {
  auto: {
    hint: "Reads the image and picks the model for it. Photographs are segmented twice and the cleaner result is kept, so they take longer.",
    title: "Choosing the best model for this image",
    detail: "Comparing candidate cutouts and keeping the one whose edges follow the picture.",
  },
  best: {
    hint: "Fast edge-aware BEN2 finishing for crisp contours, hair, glass, and difficult subjects.",
    title: "Separating subject with precision matting",
    detail: "BEN2 is running on the best available hardware and refining only the boundary.",
  },
  balanced: {
    hint: "A strong everyday option with a smaller model and shorter wait.",
    title: "Creating a balanced cutout",
    detail: "BiRefNet Lite is analyzing the complete image and refining its edges.",
  },
  fast: {
    hint: "Fastest for products and portraits on simple, smooth backgrounds.",
    title: "Creating a fast cutout",
    detail: "ISNet is finding the main subject and recovering supported details.",
  },
  birefnet: {
    hint: "A detailed alternative when Precision is too selective. Slowest on CPU.",
    title: "Running deep subject analysis",
    detail: "Full BiRefNet is working through fine detail. This may take a few minutes on CPU.",
  },
};

const state = {
  jobs: [],
  activeId: null,
  background: "transparent",
  customColor: elements.customColor.value,
  downloadFormat: "png",
  mode: "erase",
  painting: false,
  processing: false,
  refining: false,
  selectedQuality: "auto",
  strokeMode: "erase",
  strokePoints: [],
  scale: 1,
  followsFit: true,
};

function activeJob() {
  return state.jobs.find((job) => job.id === state.activeId) || null;
}

function makeId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function setStage(name) {
  elements.dropStage.classList.toggle("hidden", name !== "drop");
  elements.loadingStage.classList.toggle("hidden", name !== "loading");
  elements.editorStage.classList.toggle("hidden", name !== "editor");
}

function showStartControls(show) {
  elements.startControls.classList.toggle("hidden", !show);
  elements.resultControls.classList.toggle("hidden", show);
}

function setStatus(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("error", isError);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("This browser could not preview that image."));
    image.src = url;
  });
}

// WebKit (Safari on iPhone, iPad and Mac) creates the blob returned by toBlob()
// lazily: uploading it straight away can send an empty body, which the server
// rejects as an unreadable image, and the blob may also come back empty. Read
// the bytes out and re-wrap them in an ordinary in-memory blob, retrying once or
// twice and finally encoding through a data URL, so every caller gets a blob
// that is guaranteed to have content.
async function canvasBlob(type = "image/png", quality) {
  const materialize = async (blob) => {
    if (!blob || blob.size === 0) return null;
    const bytes = await blob.arrayBuffer();
    return bytes.byteLength > 0 ? new Blob([bytes], { type: blob.type || type }) : null;
  };
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const blob = await materialize(await new Promise((resolve) => canvas.toBlob(resolve, type, quality)));
    if (blob) return blob;
    await new Promise((resolve) => setTimeout(resolve, 40 * (attempt + 1)));
  }
  try {
    const blob = await materialize(await (await fetch(canvas.toDataURL(type, quality))).blob());
    if (blob) return blob;
  } catch {
    // fall through to the error below
  }
  throw new Error("Could not prepare the current cutout.");
}

function replaceJobResult(job, blob) {
  if (job.resultUrl) URL.revokeObjectURL(job.resultUrl);
  job.resultBlob = blob;
  job.resultUrl = URL.createObjectURL(blob);
}

function disposeJob(job) {
  if (job.sourceUrl) URL.revokeObjectURL(job.sourceUrl);
  if (job.resultUrl) URL.revokeObjectURL(job.resultUrl);
}

function jobStatus(job) {
  if (job.status === "done") return `${job.width} × ${job.height}`;
  if (job.status === "processing") return "Removing background…";
  if (job.status === "error") return job.error || "Processing failed";
  return `${formatBytes(job.file.size)} · Ready`;
}

function renderQueue() {
  elements.queueList.replaceChildren();
  elements.queueCount.textContent = state.jobs.length;
  elements.emptyQueue.classList.toggle("hidden", state.jobs.length > 0);
  elements.clearProjectButton.classList.toggle("hidden", state.jobs.length === 0);

  for (const job of state.jobs) {
    const row = document.createElement("div");
    row.className = `queue-item ${job.id === state.activeId ? "active" : ""} ${job.status === "processing" ? "processing" : ""}`;
    row.setAttribute("role", "listitem");
    row.tabIndex = 0;
    row.dataset.jobId = job.id;

    const image = document.createElement("img");
    image.className = "queue-thumb";
    image.src = job.resultUrl || job.sourceUrl;
    image.alt = "";

    const copy = document.createElement("span");
    copy.className = "queue-copy";
    const name = document.createElement("strong");
    name.textContent = job.file.name;
    const status = document.createElement("small");
    status.textContent = jobStatus(job);
    status.classList.toggle("error", job.status === "error");
    copy.append(name, status);

    const remove = document.createElement("button");
    remove.className = "queue-remove";
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${job.file.name}`);
    remove.title = "Remove from project";
    remove.textContent = "×";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      removeJob(job.id);
    });

    row.append(image, copy, remove);
    row.addEventListener("click", () => selectJob(job.id));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectJob(job.id);
      }
    });
    elements.queueList.append(row);
  }
}

function updateRemoveButton() {
  const pendingCount = state.jobs.filter((job) => job.status === "pending" || job.status === "error").length;
  elements.removeButton.disabled = state.processing || pendingCount === 0;
  elements.removeButton.textContent = pendingCount > 1 ? `Remove ${pendingCount} backgrounds` : "Remove background";
  if (pendingCount) {
    setStatus(elements.startStatus, `${pendingCount} image${pendingCount === 1 ? "" : "s"} ready · ${qualityCopy[state.selectedQuality].hint}`);
  } else {
    setStatus(elements.startStatus, "");
  }
}

function addFiles(fileList) {
  const files = [...fileList];
  let rejected = 0;
  for (const file of files) {
    if (!file.type.startsWith("image/") || file.size === 0 || file.size > MAX_FILE_BYTES) {
      rejected += 1;
      continue;
    }
    state.jobs.push({
      id: makeId(),
      file,
      sourceUrl: URL.createObjectURL(file),
      resultBlob: null,
      resultUrl: null,
      status: "pending",
      error: "",
      quality: null,
      width: 0,
      height: 0,
      undoStack: [],
      redoStack: [],
      feedbackSaved: false,
      preserveShadows: elements.preserveShadows.checked,
      background: "transparent",
      customColor: state.customColor,
    });
  }

  if (files.length && state.jobs.some((job) => job.status === "pending")) {
    state.activeId = null;
    setStage("drop");
    showStartControls(true);
  }
  renderQueue();
  updateRemoveButton();
  if (rejected) {
    setStatus(elements.startStatus, `${rejected} file${rejected === 1 ? " was" : "s were"} skipped. Use a supported image under 60 MB.`, true);
  }
}

async function responseError(response, fallback) {
  try {
    const data = await response.json();
    return data.error || fallback;
  } catch (_) {
    return fallback;
  }
}

async function loadRuntimeInfo() {
  try {
    const response = await fetch("/api/info", { cache: "no-store" });
    if (!response.ok) throw new Error("Runtime status unavailable");
    const info = await response.json();
    elements.runtimeBackend.textContent = `${info.runtime.detail} · Images stay local.`;
  } catch (_) {
    elements.runtimeBackend.textContent = "Automatic fallback enabled · Images stay local.";
  }
}

async function processJobs() {
  if (state.processing) return;
  const pending = state.jobs.filter((job) => job.status === "pending" || job.status === "error");
  if (!pending.length) return;

  state.processing = true;
  elements.removeButton.disabled = true;
  setStage("loading");
  const copy = qualityCopy[state.selectedQuality];
  elements.loadingTitle.textContent = copy.title;
  elements.loadingDetail.textContent = copy.detail;
  let completed = 0;

  for (const [index, job] of pending.entries()) {
    job.status = "processing";
    job.error = "";
    renderQueue();
    elements.loadingStep.textContent = `Image ${index + 1} of ${pending.length}`;
    elements.loadingFile.textContent = job.file.name;
    elements.progressBar.style.width = `${Math.max(4, (completed / pending.length) * 100)}%`;

    const form = new FormData();
    form.append("image", job.file);
    form.append("quality", state.selectedQuality);
    form.append("preserve_shadows", String(elements.preserveShadows.checked));

    try {
      const response = await fetch("/remove", { method: "POST", body: form });
      if (!response.ok) throw new Error(await responseError(response, "Background removal failed."));
      const blob = await response.blob();
      replaceJobResult(job, blob);
      const resultImage = await loadImage(job.resultUrl);
      job.width = resultImage.naturalWidth;
      job.height = resultImage.naturalHeight;
      const chosen = response.headers.get("X-Cutout-Quality");
      job.quality =
        state.selectedQuality === "auto" && chosen
          ? `auto:${chosen}`
          : state.selectedQuality;
      job.preserveShadows = elements.preserveShadows.checked;
      job.status = "done";
    } catch (error) {
      job.status = "error";
      job.error = error.message;
    }
    completed += 1;
    elements.progressBar.style.width = `${(completed / pending.length) * 100}%`;
    renderQueue();
  }

  state.processing = false;
  loadRuntimeInfo();
  const firstCompleted = pending.find((job) => job.status === "done") || state.jobs.find((job) => job.status === "done");
  if (firstCompleted) {
    await openJob(firstCompleted.id);
  } else {
    setStage("drop");
    showStartControls(true);
    setStatus(elements.startStatus, "None of the images could be processed. Check the queue for details.", true);
  }
  updateRemoveButton();
}

async function saveActiveResult() {
  const job = activeJob();
  if (!job || job.status !== "done" || elements.editorStage.classList.contains("hidden")) return;
  replaceJobResult(job, await canvasBlob());
}

async function selectJob(id) {
  const job = state.jobs.find((item) => item.id === id);
  if (!job || state.processing || state.refining) return;
  if (job.status === "done") {
    if (state.activeId && state.activeId !== id) await saveActiveResult();
    await openJob(id);
  } else {
    state.activeId = null;
    setStage("drop");
    showStartControls(true);
    renderQueue();
    updateRemoveButton();
    if (job.status === "error") setStatus(elements.startStatus, `${job.file.name}: ${job.error}`, true);
  }
}

async function openJob(id) {
  const job = state.jobs.find((item) => item.id === id);
  if (!job?.resultUrl) return;
  state.activeId = id;
  const [original, cutout] = await Promise.all([loadImage(job.sourceUrl), loadImage(job.resultUrl)]);

  for (const target of [elements.originalCanvas, canvas, elements.selectionCanvas]) {
    target.width = cutout.naturalWidth;
    target.height = cutout.naturalHeight;
  }
  originalContext.clearRect(0, 0, canvas.width, canvas.height);
  originalContext.drawImage(original, 0, 0, canvas.width, canvas.height);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(cutout, 0, 0, canvas.width, canvas.height);
  selectionContext.clearRect(0, 0, canvas.width, canvas.height);

  elements.resultName.textContent = job.file.name;
  elements.resultMeta.textContent = `${canvas.width} × ${canvas.height} · ${qualityLabel(job.quality)}${job.preserveShadows ? " · Shadow preserved" : ""}`;
  state.background = job.background || "transparent";
  state.customColor = job.customColor || state.customColor;
  elements.customColor.value = state.customColor;
  applyPreviewBackground(state.background);
  updateComparison(0);
  setStage("editor");
  showStartControls(false);
  state.followsFit = true;
  requestAnimationFrame(fitCanvas);
  updateHistoryButtons();
  updateFeedbackControl();
  setStatus(elements.resultStatus, "Ready to export");
  renderQueue();
}

function qualityLabel(quality) {
  if (quality && quality.startsWith("auto:")) {
    return `Automatic → ${qualityLabel(quality.slice(5))}`;
  }
  return { auto: "Automatic", best: "Precision", balanced: "Balanced", fast: "Fast", birefnet: "Deep detail" }[quality] || "Cutout";
}

function removeJob(id) {
  if (state.processing || state.refining) return;
  const index = state.jobs.findIndex((job) => job.id === id);
  if (index < 0) return;
  const [removed] = state.jobs.splice(index, 1);
  disposeJob(removed);
  if (state.activeId === id) {
    state.activeId = null;
    const nextDone = state.jobs.find((job) => job.status === "done");
    if (nextDone) openJob(nextDone.id);
    else {
      setStage("drop");
      showStartControls(true);
    }
  }
  renderQueue();
  updateRemoveButton();
}

function clearProject() {
  if (state.processing || state.refining || !state.jobs.length) return;
  for (const job of state.jobs) disposeJob(job);
  state.jobs = [];
  state.activeId = null;
  elements.fileInput.value = "";
  setStage("drop");
  showStartControls(true);
  setStatus(elements.startStatus, "");
  renderQueue();
  updateRemoveButton();
}

function fitCanvas() {
  if (!canvas.width || !canvas.height) return;
  const horizontalPadding = window.innerWidth < 560 ? 38 : 74;
  const availableWidth = Math.max(120, elements.canvasViewport.clientWidth - horizontalPadding);
  const availableHeight = Math.max(120, elements.canvasViewport.clientHeight - horizontalPadding);
  state.scale = Math.min(1, availableWidth / canvas.width, availableHeight / canvas.height);
  state.followsFit = true;
  applyScale();
  elements.canvasViewport.scrollTo({ left: 0, top: 0 });
}

function applyScale() {
  const width = Math.max(1, Math.round(canvas.width * state.scale));
  const height = Math.max(1, Math.round(canvas.height * state.scale));
  elements.canvasStack.style.width = `${width}px`;
  elements.canvasStack.style.height = `${height}px`;
  elements.zoomValue.textContent = `${Math.round(state.scale * 100)}%`;
}

function changeZoom(factor) {
  if (!canvas.width) return;
  state.scale = Math.min(4, Math.max(.04, state.scale * factor));
  state.followsFit = false;
  applyScale();
}

function updateComparison(value) {
  const percent = Number(value);
  elements.compareSlider.value = String(percent);
  elements.originalCanvas.style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
  elements.compareDivider.style.left = `${percent}%`;
  elements.compareDivider.style.display = percent > 0 && percent < 100 ? "block" : "none";
}

function applyPreviewBackground(background) {
  state.background = background;
  const job = activeJob();
  if (job) {
    job.background = background;
    job.customColor = state.customColor;
  }
  elements.canvasShell.dataset.background = background;
  elements.canvasShell.style.setProperty("--custom-background", state.customColor);
  elements.customColor.parentElement.style.setProperty("--selected-color", state.customColor);
  $$(".swatch").forEach((swatch) => {
    const selected = swatch.dataset.background === background || (swatch.classList.contains("custom") && background === "custom");
    swatch.classList.toggle("active", selected);
    swatch.setAttribute("aria-checked", String(selected));
  });
}

function setRefining(active) {
  state.refining = active;
  elements.canvasShell.classList.toggle("analyzing", active);
  $$("#toolMode button, #downloadFormat button").forEach((button) => { button.disabled = active; });
  elements.brushSize.disabled = active;
  elements.downloadButton.disabled = active;
  elements.newButton.disabled = active;
  elements.brushCursor.style.display = "none";
  updateHistoryButtons();
  updateFeedbackControl();
}

function updateHistoryButtons() {
  const job = activeJob();
  const undoCount = job?.undoStack.length || 0;
  const redoCount = job?.redoStack.length || 0;
  elements.undoButton.disabled = state.refining || undoCount === 0;
  elements.redoButton.disabled = state.refining || redoCount === 0;
  elements.historyLabel.textContent = undoCount ? `${undoCount} edit${undoCount === 1 ? "" : "s"}` : "No edits";
}

function pushHistory(stack, blob) {
  stack.push(blob);
  if (stack.length > HISTORY_LIMIT) stack.shift();
}

async function drawCanvasBlob(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const image = await loadImage(url);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function moveHistory(source, destination, message) {
  const job = activeJob();
  if (!job || state.refining || source.length === 0) return;
  setRefining(true);
  setStatus(elements.resultStatus, message);
  try {
    const currentBlob = await canvasBlob();
    const targetBlob = source[source.length - 1];
    await drawCanvasBlob(targetBlob);
    source.pop();
    pushHistory(destination, currentBlob);
    replaceJobResult(job, targetBlob);
    job.feedbackSaved = false;
    updateHistoryButtons();
  } catch (error) {
    setStatus(elements.resultStatus, error.message, true);
  } finally {
    setRefining(false);
    renderQueue();
  }
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function drawSelectionStroke(from, to) {
  const size = Number(elements.brushSize.value);
  selectionContext.save();
  selectionContext.lineWidth = size;
  selectionContext.lineCap = "round";
  selectionContext.lineJoin = "round";
  selectionContext.strokeStyle = state.strokeMode === "restore" ? "rgba(67, 224, 158, .58)" : "rgba(255, 105, 94, .58)";
  selectionContext.beginPath();
  selectionContext.moveTo(from.x, from.y);
  selectionContext.lineTo(to.x, to.y);
  selectionContext.stroke();
  selectionContext.restore();
}

function updateBrushCursor(event) {
  if (state.refining) return;
  const stackRect = elements.canvasStack.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  const displaySize = Number(elements.brushSize.value) * (canvasRect.width / canvas.width);
  elements.brushCursor.style.width = `${Math.max(6, displaySize)}px`;
  elements.brushCursor.style.height = `${Math.max(6, displaySize)}px`;
  elements.brushCursor.style.left = `${event.clientX - stackRect.left}px`;
  elements.brushCursor.style.top = `${event.clientY - stackRect.top}px`;
  elements.brushCursor.style.display = "block";
}

async function analyzeSelection(points, selectedMode) {
  const job = activeJob();
  if (!job) return;
  setRefining(true);
  setStatus(elements.resultStatus, "Analyzing selected area…");
  try {
    const currentBlob = await canvasBlob();
    const form = new FormData();
    form.append("image", job.file);
    form.append("current", currentBlob, "current.png");
    form.append("mode", selectedMode);
    form.append("brush_size", elements.brushSize.value);
    form.append("points", JSON.stringify(points.map((point) => [point.x, point.y])));
    const response = await fetch("/refine", { method: "POST", body: form });
    if (!response.ok) throw new Error(await responseError(response, "Could not analyze that selection."));
    const blob = await response.blob();
    await drawCanvasBlob(blob);
    pushHistory(job.undoStack, currentBlob);
    job.redoStack = [];
    replaceJobResult(job, blob);
    job.feedbackSaved = false;
    setStatus(elements.resultStatus, selectedMode === "restore" ? "Selection restored" : "Selection erased");
  } catch (error) {
    setStatus(elements.resultStatus, error.message, true);
  } finally {
    selectionContext.clearRect(0, 0, elements.selectionCanvas.width, elements.selectionCanvas.height);
    setRefining(false);
    updateHistoryButtons();
    renderQueue();
  }
}

function updateFeedbackControl() {
  const job = activeJob();
  const available = job?.quality === "fast" || job?.quality === "auto:fast";
  elements.feedbackControls.classList.toggle("hidden", !available);
  elements.saveFeedbackButton.disabled = !available || state.refining || job.feedbackSaved;
  elements.saveFeedbackButton.textContent = job?.feedbackSaved ? "Fast correction saved" : "Save approved Fast correction";
}

async function saveFeedback() {
  const job = activeJob();
  if (!job || !(job.quality === "fast" || job.quality === "auto:fast") || state.refining) return;
  elements.saveFeedbackButton.disabled = true;
  setStatus(elements.resultStatus, "Saving local learning example…");
  try {
    const currentBlob = await canvasBlob();
    const form = new FormData();
    form.append("image", job.file);
    form.append("current", currentBlob, "final-correction.png");
    form.append("quality", "fast");
    const response = await fetch("/feedback", { method: "POST", body: form });
    if (!response.ok) throw new Error(await responseError(response, "Could not save the local learning example."));
    const result = await response.json();
    job.feedbackSaved = true;
    setStatus(elements.resultStatus, `Saved locally · ${result.examples} example${result.examples === 1 ? "" : "s"}`);
  } catch (error) {
    setStatus(elements.resultStatus, error.message, true);
  } finally {
    updateFeedbackControl();
  }
}

async function downloadCurrent() {
  const job = activeJob();
  if (!job || state.refining) return;
  elements.downloadButton.disabled = true;
  setStatus(elements.resultStatus, `Preparing ${state.downloadFormat.toUpperCase()}…`);
  try {
    const currentBlob = await canvasBlob();
    const form = new FormData();
    form.append("current", currentBlob, "current.png");
    form.append("format", state.downloadFormat);
    form.append("filename", job.file.name);
    form.append("background", state.background);
    form.append("background_color", state.customColor);
    const response = await fetch("/export", { method: "POST", body: form });
    if (!response.ok) throw new Error(await responseError(response, `Could not create the ${state.downloadFormat.toUpperCase()}.`));
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stem = job.file.name.replace(/\.[^.]+$/, "") || "image";
    const suffix = state.background === "transparent" ? "no-bg" : "studio";
    link.href = url;
    link.download = `${stem}-${suffix}.${state.downloadFormat}`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1200);
    setStatus(elements.resultStatus, `${state.downloadFormat.toUpperCase()} downloaded`);
  } catch (error) {
    setStatus(elements.resultStatus, error.message, true);
  } finally {
    elements.downloadButton.disabled = false;
  }
}

elements.dropStage.addEventListener("click", () => elements.fileInput.click());
elements.dropStage.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});
elements.browseButton.addEventListener("click", (event) => { event.stopPropagation(); elements.fileInput.click(); });
elements.addImagesButton.addEventListener("click", () => elements.fileInput.click());
elements.newButton.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => { addFiles(elements.fileInput.files); elements.fileInput.value = ""; });

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropStage.addEventListener(eventName, (event) => { event.preventDefault(); elements.dropStage.classList.add("dragging"); });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropStage.addEventListener(eventName, (event) => { event.preventDefault(); elements.dropStage.classList.remove("dragging"); });
}
elements.dropStage.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

document.addEventListener("paste", (event) => {
  const files = [...event.clipboardData.files].filter((file) => file.type.startsWith("image/"));
  if (files.length) {
    event.preventDefault();
    addFiles(files);
  }
});

elements.qualityMode.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-quality]");
  if (!button || state.processing) return;
  state.selectedQuality = button.dataset.quality;
  $$("button[data-quality]").forEach((item) => {
    const active = item === button;
    item.classList.toggle("active", active);
    item.setAttribute("aria-checked", String(active));
  });
  elements.qualityHint.textContent = qualityCopy[state.selectedQuality].hint;
  updateRemoveButton();
});

elements.removeButton.addEventListener("click", processJobs);
elements.clearProjectButton.addEventListener("click", clearProject);
elements.fitButton.addEventListener("click", fitCanvas);
elements.zoomInButton.addEventListener("click", () => changeZoom(1.2));
elements.zoomOutButton.addEventListener("click", () => changeZoom(1 / 1.2));
elements.compareSlider.addEventListener("input", () => updateComparison(elements.compareSlider.value));

elements.backgroundOptions.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-background]");
  if (button) applyPreviewBackground(button.dataset.background);
});
elements.customColor.addEventListener("input", () => {
  state.customColor = elements.customColor.value;
  applyPreviewBackground("custom");
});

elements.toolMode.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-mode]");
  if (!button || state.refining) return;
  state.mode = button.dataset.mode;
  $$("#toolMode button").forEach((item) => item.classList.toggle("active", item === button));
  elements.brushCursor.classList.toggle("erase", state.mode === "erase");
});
elements.brushSize.addEventListener("input", () => { elements.brushValue.textContent = `${elements.brushSize.value} px`; });

canvas.addEventListener("pointerenter", updateBrushCursor);
canvas.addEventListener("pointermove", (event) => {
  updateBrushCursor(event);
  if (!state.painting) return;
  const point = canvasPoint(event);
  const previous = state.strokePoints[state.strokePoints.length - 1];
  const minimumDistance = Math.max(1, Number(elements.brushSize.value) / 10);
  if (Math.hypot(point.x - previous.x, point.y - previous.y) < minimumDistance) return;
  state.strokePoints.push(point);
  drawSelectionStroke(previous, point);
});
canvas.addEventListener("pointerleave", () => { if (!state.painting) elements.brushCursor.style.display = "none"; });
canvas.addEventListener("pointerdown", (event) => {
  if (state.refining) return;
  event.preventDefault();
  state.painting = true;
  state.strokeMode = state.mode;
  state.strokePoints = [canvasPoint(event)];
  canvas.setPointerCapture(event.pointerId);
  drawSelectionStroke(state.strokePoints[0], state.strokePoints[0]);
  setStatus(elements.resultStatus, "Release to analyze selection");
});
canvas.addEventListener("pointerup", (event) => {
  if (!state.painting) return;
  state.painting = false;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  const points = state.strokePoints;
  state.strokePoints = [];
  analyzeSelection(points, state.strokeMode);
});
canvas.addEventListener("pointercancel", (event) => {
  state.painting = false;
  state.strokePoints = [];
  selectionContext.clearRect(0, 0, elements.selectionCanvas.width, elements.selectionCanvas.height);
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
});

elements.undoButton.addEventListener("click", () => { const job = activeJob(); if (job) moveHistory(job.undoStack, job.redoStack, "Last edit undone"); });
elements.redoButton.addEventListener("click", () => { const job = activeJob(); if (job) moveHistory(job.redoStack, job.undoStack, "Last edit redone"); });
elements.saveFeedbackButton.addEventListener("click", saveFeedback);

elements.downloadFormat.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-format]");
  if (!button || state.refining) return;
  state.downloadFormat = button.dataset.format;
  $$("#downloadFormat button").forEach((item) => item.classList.toggle("active", item === button));
  elements.downloadButton.textContent = `Download ${state.downloadFormat.toUpperCase()}`;
});
elements.downloadButton.addEventListener("click", downloadCurrent);

elements.shortcutsButton.addEventListener("click", () => {
  if (typeof elements.shortcutsDialog.showModal === "function") elements.shortcutsDialog.showModal();
});

window.addEventListener("keydown", (event) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
  const key = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && key === "z") {
    event.preventDefault();
    if (event.shiftKey) elements.redoButton.click();
    else elements.undoButton.click();
  } else if ((event.ctrlKey || event.metaKey) && key === "y") {
    event.preventDefault();
    elements.redoButton.click();
  } else if (key === "+" || key === "=") {
    event.preventDefault();
    changeZoom(1.2);
  } else if (key === "-") {
    event.preventDefault();
    changeZoom(1 / 1.2);
  } else if (key === "0" && !elements.editorStage.classList.contains("hidden")) {
    event.preventDefault();
    fitCanvas();
  }
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (state.followsFit) fitCanvas(); }, 120);
});

window.addEventListener("beforeunload", () => { for (const job of state.jobs) disposeJob(job); });

renderQueue();
updateRemoveButton();
loadRuntimeInfo();
