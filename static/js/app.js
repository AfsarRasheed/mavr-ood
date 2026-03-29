/* ═══════════════════════════════════════════════════
   MAVR-OOD — Frontend JavaScript
   ═══════════════════════════════════════════════════ */

let selectedFile = null;
let stepImages = {};
let timer = null;
let seconds = 0;

// ── Initialize ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initComparison();
    checkHealth();
});

// ── Health Check ────────────────────────────────────
async function checkHealth() {
    try {
        const resp = await fetch('/api/health');
        const data = await resp.json();
        document.getElementById('gpuName').textContent = data.gpu || 'GPU';
    } catch {
        document.getElementById('gpuName').textContent = 'Connecting...';
        setTimeout(checkHealth, 3000);
    }
}

// ── Upload / Drag-Drop ──────────────────────────────
function initUpload() {
    const zone = document.getElementById('dropZone');
    const input = document.getElementById('fileInput');

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            handleFile(input.files[0]);
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('imagePreview');
        const placeholder = document.getElementById('uploadPlaceholder');
        preview.src = e.target.result;
        preview.style.display = 'block';
        placeholder.style.display = 'none';
        document.getElementById('clearBtn').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedFile = null;
    const preview = document.getElementById('imagePreview');
    const placeholder = document.getElementById('uploadPlaceholder');
    preview.src = '';
    preview.style.display = 'none';
    placeholder.style.display = 'flex';
    document.getElementById('clearBtn').style.display = 'none';
    document.getElementById('fileInput').value = '';
}

function setQuery(text) {
    document.getElementById('queryInput').value = text;
}

// ── Detection ───────────────────────────────────────
async function runDetection() {
    const query = document.getElementById('queryInput').value.trim();
    if (!selectedFile) {
        alert('Please upload an image first.');
        return;
    }
    if (!query) {
        alert('Please enter a search query.');
        return;
    }

    const btn = document.getElementById('detectBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');

    // Disable button, show loader
    btn.disabled = true;
    btnText.textContent = 'Detecting...';
    btnLoader.style.display = 'block';

    // Show + animate progress
    showProgress();
    startTimer();

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('query', query);

        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        stopTimer();
        completeProgress();

        if (data.success) {
            setTimeout(() => renderResults(data), 600);
        } else {
            alert('Detection failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        stopTimer();
        alert('Request failed: ' + err.message);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Detect';
        btnLoader.style.display = 'none';
    }
}

// ── Timer ───────────────────────────────────────────
function startTimer() {
    seconds = 0;
    const el = document.getElementById('progressTime');
    timer = setInterval(() => {
        seconds += 0.1;
        el.textContent = seconds.toFixed(1) + 's';
    }, 100);
}

function stopTimer() {
    if (timer) { clearInterval(timer); timer = null; }
}

// ── Progress Animation ─────────────────────────────
function showProgress() {
    const section = document.getElementById('progressSection');
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Reset all steps
    const steps = document.querySelectorAll('.p-step');
    steps.forEach(s => { s.classList.remove('active', 'done'); });
    document.getElementById('progressBar').style.width = '0%';

    // Animate steps one by one
    const stepTimings = [0, 2000, 4000, 6000, 7500, 9000, 10500];
    steps.forEach((step, i) => {
        setTimeout(() => {
            // Mark previous as done
            if (i > 0) steps[i - 1].classList.remove('active');
            if (i > 0) steps[i - 1].classList.add('done');
            step.classList.add('active');
            document.getElementById('progressBar').style.width =
                `${((i + 1) / steps.length) * 100}%`;
        }, stepTimings[i] || i * 1500);
    });
}

function completeProgress() {
    const steps = document.querySelectorAll('.p-step');
    steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
    document.getElementById('progressBar').style.width = '100%';
}

// ── Render Results ──────────────────────────────────
function renderResults(data) {
    const section = document.getElementById('resultsSection');
    section.style.display = 'block';

    // Comparison images
    if (data.original_image) {
        document.getElementById('compBefore').src = 'data:image/jpeg;base64,' + data.original_image;
    }
    if (data.final_overlay) {
        document.getElementById('compAfter').src = 'data:image/jpeg;base64,' + data.final_overlay;
    } else if (data.step_images) {
        // Use last step image as "after"
        const keys = Object.keys(data.step_images);
        const lastKey = keys[keys.length - 1];
        if (data.step_images[lastKey]) {
            document.getElementById('compAfter').src = 'data:image/jpeg;base64,' + data.step_images[lastKey];
        }
    }

    // Reset slider to middle
    const overlay = document.getElementById('compOverlay');
    const handle = document.getElementById('compHandle');
    overlay.style.clipPath = 'inset(0 0 0 50%)';
    handle.style.left = '50%';

    // Metrics
    document.getElementById('metricTime').textContent = data.time + 's';
    document.getElementById('metricParser').textContent = data.parsed?.parser_mode || 'rule-based';
    document.getElementById('metricSpatial').textContent = data.parsed?.spatial || 'none';
    document.getElementById('metricObject').textContent = data.parsed?.object_prompt || data.query;

    // Step images
    stepImages = data.step_images || {};
    const tabsContainer = document.getElementById('stepTabs');
    tabsContainer.innerHTML = '';
    const stepNames = Object.keys(stepImages);

    if (stepNames.length > 0) {
        stepNames.forEach((name, i) => {
            const btn = document.createElement('button');
            btn.className = 'step-tab' + (i === 0 ? ' active' : '');
            btn.textContent = formatStepName(name);
            btn.onclick = () => showStep(name, btn);
            tabsContainer.appendChild(btn);
        });
        // Show first step
        showStep(stepNames[0], tabsContainer.querySelector('.step-tab'));
    }

    // Reasoning
    document.getElementById('reasoningText').textContent =
        data.reasoning || 'No reasoning output available.';

    // Scroll to results
    setTimeout(() => {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function formatStepName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase())
        .replace('Gdino', 'GDINO')
        .replace('Clip', 'CLIP')
        .replace('Sam', 'SAM');
}

function showStep(stepKey, clickedTab) {
    // Update tabs
    document.querySelectorAll('.step-tab').forEach(t => t.classList.remove('active'));
    if (clickedTab) clickedTab.classList.add('active');

    // Show image
    const img = document.getElementById('stepImage');
    if (stepImages[stepKey]) {
        img.src = 'data:image/jpeg;base64,' + stepImages[stepKey];
    } else {
        img.src = '';
    }
}

// ── Image Comparison Slider ─────────────────────────
function initComparison() {
    const container = document.getElementById('comparisonContainer');
    if (!container) return;

    let dragging = false;

    const onMove = (clientX) => {
        if (!dragging) return;
        const rect = container.getBoundingClientRect();
        let x = (clientX - rect.left) / rect.width;
        x = Math.max(0.05, Math.min(0.95, x));

        const pct = (x * 100);
        document.getElementById('compOverlay').style.clipPath = `inset(0 0 0 ${pct}%)`;
        document.getElementById('compHandle').style.left = pct + '%';
    };

    container.addEventListener('mousedown', (e) => {
        dragging = true;
        onMove(e.clientX);
    });
    document.addEventListener('mousemove', (e) => onMove(e.clientX));
    document.addEventListener('mouseup', () => { dragging = false; });

    // Touch support
    container.addEventListener('touchstart', (e) => {
        dragging = true;
        onMove(e.touches[0].clientX);
    });
    document.addEventListener('touchmove', (e) => {
        if (dragging) onMove(e.touches[0].clientX);
    });
    document.addEventListener('touchend', () => { dragging = false; });
}
