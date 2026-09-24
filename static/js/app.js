let currentModality = 'forensics';
let currentFile = null;
let currentScanId = null;

function switchTab(modality) {
  currentModality = modality;
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');

  const dropIcon = document.getElementById('dropIcon');
  const dropText = document.getElementById('dropText');
  const fileMeta = document.getElementById('fileMeta');

  if (modality === 'forensics') {
    dropIcon.innerText = '🔬';
    dropText.innerText = 'Select or drop an Image (.jpg, .png, .webp)';
    fileMeta.innerText = 'Multi-Signal Pipeline: ELA Analysis • 2D FFT Spectrum • Deepfake CNN • Grad-CAM';
  } else if (modality === 'image') {
    dropIcon.innerText = '🖼️';
    dropText.innerText = 'Select or drop an Image (.jpg, .png)';
    fileMeta.innerText = 'Quick scan: facial blending, compression artifacts & 2D FFT';
  } else if (modality === 'video') {
    dropIcon.innerText = '🎬';
    dropText.innerText = 'Select or drop a Video (.mp4)';
    fileMeta.innerText = 'Analyzes frame delta jitter & lip-sync variance';
  } else if (modality === 'audio') {
    dropIcon.innerText = '🎙️';
    dropText.innerText = 'Select or drop an Audio file (.wav, .mp3)';
    fileMeta.innerText = 'Analyzes spectral flux & synthetic voice clone harmonics';
  }
  currentFile = null;
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;
  currentFile = file;
  document.getElementById('dropText').innerText = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
}

// Preset generators for quick live demonstration
function loadPreset(type) {
  if (type === 'fake_image' || type === 'real_image') {
    switchTab('image');
    document.querySelectorAll('.tab-btn')[0].classList.add('active');
    
    // Create dummy canvas image
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = type === 'fake_image' ? '#2c1654' : '#1e3a5f';
    ctx.fillRect(0, 0, 300, 300);
    
    // Draw face silhouette
    ctx.fillStyle = '#f6d8ae';
    ctx.beginPath();
    ctx.arc(150, 140, 80, 0, Math.PI * 2);
    ctx.fill();
    
    // Eyes
    ctx.fillStyle = '#111';
    ctx.beginPath();
    ctx.arc(120, 130, 8, 0, Math.PI * 2);
    ctx.arc(180, 130, 8, 0, Math.PI * 2);
    ctx.fill();
    
    // Mouth
    ctx.strokeStyle = '#a33';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(150, 175, 25, 0.2 * Math.PI, 0.8 * Math.PI);
    ctx.stroke();

    if (type === 'fake_image') {
      // Add fake blending noise artifact
      ctx.fillStyle = 'rgba(255, 0, 100, 0.35)';
      ctx.fillRect(100, 150, 100, 40);
    }

    canvas.toBlob((blob) => {
      currentFile = new File([blob], `${type}.jpg`, { type: 'image/jpeg' });
      document.getElementById('dropText').innerText = `Preset Loaded: ${type}.jpg`;
      runAnalysis();
    }, 'image/jpeg');

  } else if (type === 'fake_audio') {
    switchTab('audio');
    document.querySelectorAll('.tab-btn')[2].classList.add('active');
    // Synthetic 1-second sine wave audio
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const sampleRate = 44100;
    const buffer = audioCtx.createBuffer(1, sampleRate, sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < sampleRate; i++) {
      data[i] = Math.sin(2 * Math.PI * 440 * (i / sampleRate)) * (i % 2000 < 500 ? 0.8 : 0.05);
    }
    // Convert to wav blob or generic audio
    const blob = new Blob([new Uint8Array(44100 * 2)], { type: 'audio/wav' });
    currentFile = new File([blob], 'cloned_voice_sample.wav', { type: 'audio/wav' });
    document.getElementById('dropText').innerText = 'Preset Loaded: cloned_voice_sample.wav';
    runAnalysis();

  } else if (type === 'fake_video') {
    switchTab('video');
    document.querySelectorAll('.tab-btn')[1].classList.add('active');
    const blob = new Blob(["fake video binary stream data"], { type: 'video/mp4' });
    currentFile = new File([blob], 'face_swap_demo.mp4', { type: 'video/mp4' });
    document.getElementById('dropText').innerText = 'Preset Loaded: face_swap_demo.mp4';
    runAnalysis();
  }
}

async function runAnalysis() {
  if (!currentFile) {
    alert("Please select a file or click one of the Quick Demo Presets first!");
    return;
  }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.innerText = "⏳ Processing Forensic Pipeline...";

  const formData = new FormData();
  formData.append('file', currentFile);

  const endpoint = (currentModality === 'forensics') ? '/api/forensics/analyze' : `/api/analyze/${currentModality}`;

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`API error (${res.status}): ${errText}`);
    }

    const data = await res.json();
    displayResults(data);
    loadRecentScans();
  } catch (err) {
    alert("Inference failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerText = "🚀 Run Inference Pipeline";
  }
}

function displayResults(data) {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'block';
  document.getElementById('procTimeBadge').innerText = `Completed in ${data.processing_time_ms}ms`;

  const banner = document.getElementById('verdictBanner');
  const verdictText = document.getElementById('verdictText');
  const verdictPercent = document.getElementById('verdictPercent');
  const confidenceFill = document.getElementById('confidenceFill');
  const confidenceLabel = document.getElementById('confidenceLabel');
  const explanationText = document.getElementById('explanationText');
  const forensicGrid = document.getElementById('forensicVisualGrid');
  const heatmapSec = document.getElementById('heatmapSection');
  const tagList = document.getElementById('tagList');

  // Check if this is the structured Image Forensics response
  if (data.prediction && data.ela && data.fft) {
    currentScanId = data.scan_id;
    const isFake = (data.prediction.label === 'deepfake');
    const conf = data.prediction.confidence;
    const pct = (conf * 100).toFixed(1);

    verdictPercent.innerText = `${pct}%`;
    confidenceLabel.innerText = conf.toFixed(4);
    confidenceFill.style.width = `${pct}%`;

    if (isFake) {
      banner.className = 'verdict-banner verdict-fake';
      verdictText.innerText = `🚨 DEEPFAKE DETECTED`;
      confidenceFill.style.background = 'var(--accent-red)';
    } else {
      banner.className = 'verdict-banner verdict-real';
      verdictText.innerText = `✅ AUTHENTIC MEDIA`;
      confidenceFill.style.background = 'var(--accent-green)';
    }

    explanationText.innerHTML = `<strong>Primary Neural Model:</strong> ${data.prediction.model_architecture} classified image as <strong>${data.prediction.label.toUpperCase()}</strong> with ${pct}% confidence.<br><strong>Forensic Corroboration:</strong> ELA compression error (mean ${data.ela.mean_error}, high-err ratio ${(data.ela.high_error_ratio * 100).toFixed(1)}%) and 2D FFT spectral decay (ratio ${data.fft.high_frequency_ratio}).`;

    // Populate and show 4-card forensic visual grid
    forensicGrid.style.display = 'block';
    heatmapSec.style.display = 'none';

    document.getElementById('forensicOriginalImg').src = data.original_image_url;
    document.getElementById('forensicElaImg').src = data.ela.image_url;
    document.getElementById('elaMean').innerText = data.ela.mean_error;
    document.getElementById('elaMax').innerText = data.ela.max_error;
    document.getElementById('elaHighRatio').innerText = `${(data.ela.high_error_ratio * 100).toFixed(2)}%`;

    document.getElementById('forensicFftImg').src = data.fft.image_url;
    document.getElementById('fftMean').innerText = data.fft.mean_magnitude;
    document.getElementById('fftMax').innerText = data.fft.max_magnitude;
    document.getElementById('fftRatio').innerText = data.fft.high_frequency_ratio;

    if (data.gradcam && data.gradcam.available) {
      document.getElementById('forensicGradcamImg').src = data.gradcam.heatmap_url;
      document.getElementById('forensicOverlayImg').src = data.gradcam.overlay_url;
      document.getElementById('gradcamLayer').innerText = data.gradcam.target_layer || 'layer4';
      document.getElementById('gradcamTarget').innerText = data.gradcam.target_class || 'deepfake';
    } else {
      document.getElementById('gradcamLayer').innerText = 'Unavailable';
      document.getElementById('gradcamTarget').innerText = 'N/A';
    }

    tagList.innerHTML = '';
    const forensicTags = [
      `Model: ${data.prediction.model_architecture}`,
      `Confidence: ${pct}%`,
      `ELA Quality: ${data.ela.quality}`,
      `2D FFT Spectrum: Active`,
      `Grad-CAM: ${data.gradcam.target_layer || 'layer4'} Backprop`
    ];
    if (data.warnings && data.warnings.length > 0) {
      data.warnings.forEach(w => forensicTags.push(`⚠️ ${w}`));
    }
    forensicTags.forEach(t => {
      const span = document.createElement('span');
      span.className = 'tag';
      span.innerText = t;
      tagList.appendChild(span);
    });

    return;
  }

  // Fallback for legacy modality responses (video, audio, quick-scan)
  forensicGrid.style.display = 'none';
  currentScanId = data.id;

  const pct = (data.confidence_score * 100).toFixed(1);
  verdictPercent.innerText = `${pct}%`;
  confidenceLabel.innerText = data.confidence_score.toFixed(4);
  confidenceFill.style.width = `${pct}%`;

  if (data.is_fake) {
    banner.className = 'verdict-banner verdict-fake';
    verdictText.innerText = `🚨 ${data.verdict}`;
    confidenceFill.style.background = 'var(--accent-red)';
  } else {
    banner.className = 'verdict-banner verdict-real';
    verdictText.innerText = `✅ ${data.verdict}`;
    confidenceFill.style.background = 'var(--accent-green)';
  }

  explanationText.innerText = data.explanation;

  const heatmapImg = document.getElementById('heatmapImg');
  if (data.heatmap_url) {
    heatmapSec.style.display = 'block';
    heatmapImg.src = data.heatmap_url;
  } else {
    heatmapSec.style.display = 'none';
  }

  // Tags
  const tagList = document.getElementById('tagList');
  tagList.innerHTML = '';
  const tags = data.detected_features || [
    `Temporal Flicker: ${(data.temporal_flicker_score * 100 || 0).toFixed(1)}%`,
    `Spectral Flux: ${(data.spectral_flux_score * 100 || 0).toFixed(1)}%`
  ];

  tags.forEach(t => {
    const span = document.createElement('span');
    span.className = 'tag';
    span.innerText = `• ${t}`;
    tagList.appendChild(span);
  });
}

async function sendFeedback(verdict) {
  if (!currentScanId) return;
  try {
    const res = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        analysis_id: currentScanId,
        user_verdict: verdict,
        notes: `Validated in live demo by evaluator.`
      })
    });
    if (res.ok) {
      alert(`Feedback recorded: ${verdict}. Logged to SQLite active learning dataset.`);
      loadRecentScans();
    }
  } catch (err) {
    console.error(err);
  }
}

async function loadRecentScans() {
  try {
    const res = await fetch('/api/scans/recent');
    const scans = await res.json();
    const tbody = document.getElementById('historyTableBody');
    tbody.innerHTML = '';

    if (scans.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No scans recorded yet.</td></tr>';
      return;
    }

    scans.forEach(s => {
      const tr = document.createElement('tr');
      const timeStr = new Date(s.created_at).toLocaleTimeString();
      const dotClass = s.is_fake ? 'dot-red' : 'dot-green';
      
      tr.innerHTML = `
        <td style="font-family: monospace; color: var(--accent-cyan);">#${s.id}</td>
        <td><span class="tag" style="text-transform: uppercase;">${s.media_type}</span></td>
        <td style="max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${s.filename}</td>
        <td><span class="status-dot ${dotClass}"></span>${s.verdict}</td>
        <td>${(s.confidence_score * 100).toFixed(1)}%</td>
        <td>${s.processing_time_ms}ms</td>
        <td><span style="color: #a78bfa; font-size: 11px;">${s.feedback_verdict || 'Unreviewed'}</span></td>
        <td style="color: var(--text-muted); font-size: 11px;">${timeStr}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load scans:", err);
  }
}

// Initial fetch on page load
window.addEventListener('DOMContentLoaded', () => {
  loadRecentScans();
});
