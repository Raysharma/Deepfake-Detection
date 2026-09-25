// DeepShield AI Frontend Engine
const BACKEND_API_URL = "http://127.0.0.1:8000/api/v1/upload";

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const dropContent = document.getElementById("dropContent");
const previewZone = document.getElementById("previewZone");
const mediaContainer = document.getElementById("mediaContainer");
const previewName = document.getElementById("previewName");
const previewSize = document.getElementById("previewSize");
const btnClear = document.getElementById("btnClear");
const btnAnalyze = document.getElementById("btnAnalyze");

const resultsSection = document.getElementById("resultsSection");

// Evidence Drawer Elements
const btnToggleEvidence = document.getElementById("btnToggleEvidence");
const evidenceDrawer = document.getElementById("evidenceDrawer");
const evidenceChevron = document.getElementById("evidenceChevron");
const evidenceFaceImg = document.getElementById("evidenceFaceImg");
const evidenceAudioImg = document.getElementById("evidenceAudioImg");
const syncCanvas = document.getElementById("syncCanvas");
const syncRadarBadge = document.getElementById("syncRadarBadge");

let selectedFile = null;

// Evidence Drawer Toggle
if (btnToggleEvidence) {
    btnToggleEvidence.addEventListener("click", () => {
        const isHidden = evidenceDrawer.classList.contains("hidden");
        if (isHidden) {
            evidenceDrawer.classList.remove("hidden");
            evidenceChevron.classList.add("rotate");
        } else {
            evidenceDrawer.classList.add("hidden");
            evidenceChevron.classList.remove("rotate");
        }
    });
}

// File Selection Handlers
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

btnClear.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
});

function handleFile(file) {
    if (!file) return;
    selectedFile = file;

    previewName.textContent = file.name;
    previewSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";

    const overlayHtml = `
    <div class="scanner-overlay hidden" id="scannerOverlay">
      <div class="laser-beam"></div>
      <div class="scanner-hud">
        <div class="hud-spinner"></div>
        <div class="hud-text" id="hudStatus">Initializing Neural Models...</div>
        <div class="hud-sub">Multimodal Analysis in Progress</div>
      </div>
    </div>
  `;
    mediaContainer.innerHTML = overlayHtml;

    const fileUrl = URL.createObjectURL(file);

    if (file.type.startsWith("video/")) {
        const video = document.createElement("video");
        video.src = fileUrl;
        video.controls = true;
        mediaContainer.appendChild(video);
    } else if (file.type.startsWith("image/")) {
        const img = document.createElement("img");
        img.src = fileUrl;
        mediaContainer.appendChild(img);
    } else if (file.type.startsWith("audio/")) {
        const audio = document.createElement("audio");
        audio.src = fileUrl;
        audio.controls = true;
        mediaContainer.appendChild(audio);
    }

    dropContent.classList.add("hidden");
    previewZone.classList.remove("hidden");
    btnAnalyze.disabled = false;
    resultsSection.classList.add("hidden");
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    mediaContainer.innerHTML = "";
    previewZone.classList.add("hidden");
    dropContent.classList.remove("hidden");
    btnAnalyze.disabled = true;
    resultsSection.classList.add("hidden");
}

// Execute Forensics Analysis
btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = `<span>Forensics Scan Active...</span>`;

    const scannerOverlay = document.getElementById("scannerOverlay");
    const hudStatus = document.getElementById("hudStatus");
    if (scannerOverlay) scannerOverlay.classList.remove("hidden");

    resultsSection.classList.add("hidden");

    const steps = [
        "🔍 Modality 1: Analyzing Spatial Compression (ELA / 2D FFT)...",
        "🎙️ Modality 2: Extracting Acoustic Mel-Spectrogram (CNN)...",
        "👄 Modality 3: Evaluating Lip-Sync Visemes (SyncNet)...",
        "⚖️ Computing Tri-Fusion Security Verdict..."
    ];
    let stepIdx = 0;
    if (hudStatus) hudStatus.textContent = steps[0];
    const ticker = setInterval(() => {
        stepIdx = (stepIdx + 1) % steps.length;
        if (hudStatus) hudStatus.textContent = steps[stepIdx];
    }, 1200);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch(BACKEND_API_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();
        const result = data.result || data;

        clearInterval(ticker);
        if (scannerOverlay) scannerOverlay.classList.add("hidden");
        renderResults(result);
    } catch (err) {
        clearInterval(ticker);
        if (scannerOverlay) scannerOverlay.classList.add("hidden");
        alert("Connection error: Ensure FastAPI server is running on http://127.0.0.1:8000\n\nDetails: " + err.message);
    } finally {
        btnAnalyze.disabled = false;
        btnAnalyze.innerHTML = `
      <span class="btn-text">Execute Forensics Analysis</span>
      <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="5 3 19 12 5 21 5 3"/>
      </svg>
    `;
    }
});

function renderResults(data) {
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth" });

    const isFake = data.overall_verdict === "FAKE";
    const banner = document.getElementById("verdictBanner");
    banner.className = `verdict-banner ${isFake ? "fake" : "real"}`;

    document.getElementById("verdictTitle").textContent = isFake ? "MANIPULATION DETECTED" : "AUTHENTIC MEDIA";
    document.getElementById("verdictSummary").textContent = isFake
        ? "High probability of synthetic facial manipulation, dubbed audio, or desynchronization."
        : "Biometric and acoustic signals verify this media as genuine human recording.";

    const fakePercent = Math.round((data.overall_fake_probability || 0) * 100);
    document.getElementById("verdictScore").textContent = `${fakePercent}%`;
    document.getElementById("verdictTime").textContent = `Processing Time: ${data.processing_time_sec || 0}s`;

    // 1. Image Modality
    const img = data.image_analysis || {};
    document.getElementById("imgFakeProb").textContent = `${Math.round((img.fake_probability || 0) * 100)}%`;
    document.getElementById("imgFrames").textContent = img.frames_analyzed || 0;
    setPill("pillImage", img.is_fake);
    setFindings("imgFindings", img.detected_artifacts);

    // 2. Audio Modality
    const aud = data.audio_analysis || {};
    document.getElementById("audFakeProb").textContent = `${Math.round((aud.fake_probability || 0) * 100)}%`;
    document.getElementById("audDuration").textContent = `${aud.audio_duration_sec || 0}s`;
    setPill("pillAudio", aud.is_fake);
    setFindings("audFindings", aud.detected_anomalies);

    // 3. Lip-Sync Modality (SyncNet)
    const sync = data.video_sync_analysis || {};
    document.getElementById("syncConfidence").textContent = `${Math.round((sync.sync_confidence || 0) * 100)}%`;
    document.getElementById("syncOffset").textContent = `${sync.offset_frames || 0} frames`;
    setPill("pillSync", sync.is_fake);
    setFindings("syncFindings", sync.detected_anomalies);

    // 4. Render Deep Forensic Evidence
    renderForensicEvidence(img, aud, sync);
}

function renderForensicEvidence(img, aud, sync) {
    const timestamp = Date.now();

    // 1. Facial Grad-CAM Heatmap
    if (img.heatmap_path) {
        const cleanPath = img.heatmap_path.replace(/\\/g, "/");
        evidenceFaceImg.src = `/${cleanPath}?t=${timestamp}`;
        document.getElementById("evidenceFaceCard").classList.remove("hidden");
    } else {
        evidenceFaceImg.src = "https://placehold.co/300x200/121826/00f0ff?text=No+Face+Heatmap";
    }

    // 2. Audio Mel-Spectrogram Heatmap
    if (aud.spectrogram_heatmap_path) {
        const cleanAudioPath = aud.spectrogram_heatmap_path.replace(/\\/g, "/");
        evidenceAudioImg.src = `/${cleanAudioPath}?t=${timestamp}`;
        document.getElementById("evidenceAudioCard").classList.remove("hidden");
    } else {
        evidenceAudioImg.src = "https://placehold.co/300x200/121826/00f0ff?text=Audio+N/A";
    }

    // 3. SyncNet Cross-Correlation Curve Radar (Canvas)
    const offset = sync.offset_frames || 0;
    const isSyncFake = sync.is_fake || false;
    syncRadarBadge.textContent = `${offset > 0 ? "+" : ""}${offset} frames delay`;
    drawSyncCurve(offset, isSyncFake);
}

// Draws the temporal cross-correlation curve across the ±10 frame search window
function drawSyncCurve(offset, isFake) {
    if (!syncCanvas) return;
    const ctx = syncCanvas.getContext("2d");
    const w = syncCanvas.width;
    const h = syncCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Background Grid Lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    for (let x = 30; x < w; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 10);
        ctx.lineTo(x, h - 25);
        ctx.stroke();
    }

    // Center Zero Axis
    const centerX = w / 2;
    ctx.strokeStyle = "rgba(0, 240, 255, 0.25)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(centerX, 10);
    ctx.lineTo(centerX, h - 25);
    ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px JetBrains Mono";
    ctx.fillText("-10f", 10, h - 10);
    ctx.fillText("0 (Perfect Sync)", centerX - 45, h - 10);
    ctx.fillText("+10f", w - 35, h - 10);

    // Peak X coordinate based on offset (-10 to +10 mapped to [30, w-30])
    const peakX = centerX + (offset / 10) * (centerX - 35);
    const color = isFake ? "#ff3366" : "#00e676";

    // Draw Bell Curve
    ctx.beginPath();
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;

    for (let x = 20; x <= w - 20; x += 2) {
        const dist = (x - peakX) / 28;
        const yVal = isFake ? 0.35 * Math.exp(-dist * dist) : Math.exp(-dist * dist);
        const y = (h - 35) - yVal * (h - 60);
        if (x === 20) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Mark Peak Point
    ctx.fillStyle = "#ffffff";
    const peakY = (h - 35) - (isFake ? 0.35 : 1.0) * (h - 60);
    ctx.beginPath();
    ctx.arc(peakX, peakY, 4, 0, Math.PI * 2);
    ctx.fill();
}

function setPill(id, isFake) {
    const el = document.getElementById(id);
    if (isFake) {
        el.textContent = "FLAGGED";
        el.className = "status-pill fail";
    } else {
        el.textContent = "PASSED";
        el.className = "status-pill pass";
    }
}

function setFindings(id, list) {
    const el = document.getElementById(id);
    el.innerHTML = "";
    if (!list || list.length === 0) {
        el.innerHTML = "<li>No anomalies found</li>";
        return;
    }
    list.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = `• ${item.replace(/_/g, " ")}`;
        el.appendChild(li);
    });
}
