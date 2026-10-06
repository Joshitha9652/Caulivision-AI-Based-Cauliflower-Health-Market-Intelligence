const fileInput = document.getElementById("file-input");
const camBtn = document.getElementById("cam-btn");
const shotBtn = document.getElementById("shot-btn");
const runBtn = document.getElementById("run-btn");
const live = document.getElementById("live");
const preview = document.getElementById("preview");
const snap = document.getElementById("snap");
const camStatus = document.getElementById("cam-status");
const diseaseCard = document.getElementById("disease-card");
const qualityCard = document.getElementById("quality-card");
const adviceCard = document.getElementById("advice-card");

let blob = null;
let stream = null;

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  stopCam();
  blob = file;
  showPreview(URL.createObjectURL(file));
  runBtn.disabled = false;
});

camBtn.addEventListener("click", async () => {
  camStatus.textContent = "";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    live.srcObject = stream;
    live.hidden = false;
    preview.hidden = true;
    shotBtn.disabled = false;
    camStatus.textContent = "Camera is on. Frame the cauliflower, then capture.";
  } catch (err) {
    camStatus.textContent = "Camera permission denied or not available. Upload an image instead.";
  }
});

shotBtn.addEventListener("click", async () => {
  const w = live.videoWidth || 640;
  const h = live.videoHeight || 480;
  snap.width = w;
  snap.height = h;
  snap.getContext("2d").drawImage(live, 0, 0, w, h);
  blob = await new Promise((resolve) => snap.toBlob(resolve, "image/jpeg", 0.9));
  showPreview(URL.createObjectURL(blob));
  stopCam();
  runBtn.disabled = false;
  camStatus.textContent = "Photo captured. Run prediction.";
});

runBtn.addEventListener("click", async () => {
  if (!blob) return;
  runBtn.disabled = true;
  diseaseCard.innerHTML = "<h2>Disease predict</h2><p>Reading the curd…</p>";
  qualityCard.innerHTML = "<h2>Quality predict</h2><p>Scoring grade…</p>";
  adviceCard.hidden = true;
  const body = new FormData();
  body.append("image", blob, "capture.jpg");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    renderDisease(data.disease);
    renderQuality(data.quality);
    renderAdvice(data.advice);
  } catch (err) {
    diseaseCard.innerHTML = `<h2>Disease predict</h2><p>${err.message}</p>`;
    qualityCard.innerHTML = "<h2>Quality predict</h2><p>No quality result because analysis stopped.</p>";
  } finally {
    runBtn.disabled = false;
  }
});

function showPreview(url) {
  preview.src = url;
  preview.hidden = false;
  live.hidden = true;
}

function stopCam() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  live.srcObject = null;
  shotBtn.disabled = true;
}

function badgeClass(text) {
  const t = String(text).toLowerCase();
  if (t.includes("healthy") || t.includes("grade a") || t === "none" || t === "low") return "badge";
  if (t.includes("grade b") || t === "medium") return "badge warn";
  return "badge bad";
}

function renderDisease(d) {
  diseaseCard.innerHTML = `
    <h2>Disease predict</h2>
    <p><span class="${badgeClass(d.name)}">${d.name}</span>
       <span class="${badgeClass(d.severity)}">${d.severity}</span></p>
    <p>${d.summary}</p>
    <div class="metric"><span>Condition</span><strong>${d.condition}</strong></div>
    <div class="metric"><span>Spots</span><strong>${d.spots_count}</strong></div>
    <div class="metric"><span>Confidence</span><strong>${Math.round(d.confidence * 100)}%</strong></div>
    <div class="metric"><span>Cream / white</span><strong>${d.signals.white_cream}%</strong></div>
    <div class="metric"><span>Yellow</span><strong>${d.signals.yellow}%</strong></div>
    <div class="metric"><span>Brown / dark</span><strong>${d.signals.brown}% / ${d.signals.dark}%</strong></div>
  `;
}

function renderQuality(q) {
  qualityCard.innerHTML = `
    <h2>Quality predict</h2>
    <p><span class="${badgeClass(q.grade)}">${q.grade}</span>
       <span class="${badgeClass(q.condition)}">${q.condition}</span></p>
    <p>${q.summary}</p>
    <div class="metric"><span>Color score</span><strong>${q.color_score} / 10</strong></div>
    <div class="metric"><span>Firmness</span><strong>${q.firmness_score} / 10</strong></div>
    <div class="metric"><span>Leaf condition</span><strong>${q.leaf_condition_score} / 10</strong></div>
    <div class="metric"><span>Spots</span><strong>${q.spots_count}</strong></div>
    <div class="metric"><span>Confidence</span><strong>${Math.round(q.confidence * 100)}%</strong></div>
  `;
}

function list(id, items) {
  document.getElementById(id).innerHTML = (items || []).map((x) => `<li>${x}</li>`).join("");
}

function renderAdvice(a) {
  adviceCard.hidden = false;
  document.getElementById("advice-source").textContent =
    a.source === "llm"
      ? "Advice generated by the language model from your disease + quality scores."
      : "Offline agronomy pack used (add GROQ_API_KEY or OPENAI_API_KEY in a .env file to use an LLM).";
  list("adv-precautions", a.precautions);
  list("adv-measures", a.measures);
  list("adv-soil", a.black_soil);
  document.getElementById("adv-pesticides").innerHTML = (a.pesticides || [])
    .map(
      (p) =>
        `<tr><td>${p.name || ""}</td><td>${p.use_for || ""}</td><td>${p.dose || ""}</td><td>${p.note || ""}</td></tr>`
    )
    .join("");
}
