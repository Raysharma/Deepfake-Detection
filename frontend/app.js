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

const scanningCard = document.getElementById("scanningCard");
const resultsSection = document.getElementById("resultsSection");

let selectedFile = null;

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

    mediaContainer.innerHTML = "";
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
    scanningCard.classList.add("hidden");
}

// Execute Forensics Analysis
btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;

    btnAnalyze.disabled = true;
    scanningCard.classList.remove("hidden");
    resultsSection.classList.add("hidden");

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

        renderResults(result);
    } catch (err) {
        alert("Connection error: Ensure FastAPI server is running on http://127.0.0.1:8000\n\nDetails: " + err.message);
    } finally {
        scanningCard.classList.add("hidden");
        btnAnalyze.disabled = false;
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
