/* ═══════════════════════════════════════════════════
   MAVR-OOD — Frontend JavaScript
   Handles both Text-Guided + OOD Detection tabs
   ═══════════════════════════════════════════════════ */

let selectedFile = null;
let oodFile = null;
let gtFile = null;
let stepImages = {};
let timer = null;
let seconds = 0;

// ── Initialize ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initUpload('dropZone', 'fileInput', 'imagePreview', 'uploadPlaceholder', 'clearBtn', (f) => selectedFile = f);
    initUpload('oodDropZone', 'oodFileInput', 'oodImagePreview', 'oodUploadPlaceholder', null, (f) => oodFile = f);
    initUpload('gtDropZone', 'gtFileInput', 'gtImagePreview', 'gtUploadPlaceholder', null, (f) => gtFile = f);
    initComparison();
    checkHealth();
});

// ── Tab Switching ───────────────────────────────────
function switchTab(tab) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

    if (tab === 'guided') {
        document.getElementById('tabGuided').classList.add('active');
        document.getElementById('guidedTab').classList.add('active');
    } else {
        document.getElementById('tabOod').classList.add('active');
        document.getElementById('oodTab').classList.add('active');
    }
}

// ── Health Check ────────────────────────────────────
async function checkHealth() {
    try {
        const resp = await fetch('/api/health');
        const data = await resp.json();
        const el = document.getElementById('gpuName');
        el.textContent = data.gpu || 'GPU';
    } catch {
        document.getElementById('gpuName').textContent = 'Connecting...';
        setTimeout(checkHealth, 3000);
    }
}

// ── Generic Upload / Drag-Drop ──────────────────────
function initUpload(zoneId, inputId, previewId, placeholderId, clearBtnId, callback) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) loadFile(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files.length > 0) loadFile(input.files[0]); });

    function loadFile(file) {
        if (!file.type.startsWith('image/')) return;
        callback(file);
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById(previewId);
            const placeholder = document.getElementById(placeholderId);
            preview.src = e.target.result;
            preview.style.display = 'block';
            placeholder.style.display = 'none';
            if (clearBtnId) document.getElementById(clearBtnId).style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

function clearImage() {
    selectedFile = null;
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('uploadPlaceholder').style.display = 'flex';
    document.getElementById('clearBtn').style.display = 'none';
    document.getElementById('fileInput').value = '';
}

function setQuery(text) {
    document.getElementById('queryInput').value = text;
}

// ── Text-Guided Detection ───────────────────────────
async function runDetection() {
    const query = document.getElementById('queryInput').value.trim();
    if (!selectedFile) return showError('Please upload an image first.');
    if (!query) return showError('Please enter a search query.');
    hideError();

    const btn = document.getElementById('detectBtn');
    setBtnLoading(btn, true, 'Detecting...');
    showProgress();
    startTimer();

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('query', query);

        const response = await fetch('/api/detect', { method: 'POST', body: formData });
        const data = await safeJson(response);
        stopTimer();
        completeProgress();

        if (data.success) {
            setTimeout(() => renderResults(data), 500);
        } else {
            showError('Detection failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        stopTimer();
        showError('Request failed: ' + err.message);
    } finally {
        setBtnLoading(btn, false, 'Detect');
    }
}

// ── OOD Detection ───────────────────────────────────
async function runOodDetection() {
    if (!oodFile) return alert('Please upload a road scene image.');

    const btn = document.getElementById('oodDetectBtn');
    setBtnLoading(btn, true, 'Analyzing...');

    try {
        const formData = new FormData();
        formData.append('image', oodFile);
        if (gtFile) formData.append('gt_mask', gtFile);

        const response = await fetch('/api/ood_detect', { method: 'POST', body: formData });
        const data = await safeJson(response);

        if (data.success) {
            renderOodResults(data);
        } else {
            alert('OOD detection failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        alert('OOD request failed: ' + err.message);
    } finally {
        setBtnLoading(btn, false, 'Run OOD Detection');
    }
}

// ── Safe JSON Parse ─────────────────────────────────
async function safeJson(response) {
    const text = await response.text();
    try {
        return JSON.parse(text);
    } catch {
        // Server returned non-JSON (HTML error page)
        const match = text.match(/<title>(.*?)<\/title>/i);
        const hint = match ? match[1] : text.substring(0, 120);
        throw new Error('Server error: ' + hint);
    }
}

// ── Error Handling ──────────────────────────────────
function showError(msg) {
    const el = document.getElementById('errorBanner');
    el.textContent = msg;
    el.style.display = 'block';
}
function hideError() {
    document.getElementById('errorBanner').style.display = 'none';
}

// ── Button State ────────────────────────────────────
function setBtnLoading(btn, loading, text) {
    btn.disabled = loading;
    btn.querySelector('.btn-text').textContent = text;
    btn.querySelector('.btn-loader').style.display = loading ? 'block' : 'none';
}

// ── Timer ───────────────────────────────────────────
function startTimer() {
    seconds = 0;
    const el = document.getElementById('progressTime');
    timer = setInterval(() => { seconds += 0.1; el.textContent = seconds.toFixed(1) + 's'; }, 100);
}
function stopTimer() { if (timer) { clearInterval(timer); timer = null; } }

// ── Progress Animation ─────────────────────────────
function showProgress() {
    const section = document.getElementById('progressSection');
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth', block: 'center' });

    const steps = document.querySelectorAll('.p-step');
    steps.forEach(s => s.classList.remove('active', 'done'));
    document.getElementById('progressBar').style.width = '0%';

    const timings = [0, 2500, 5000, 7500, 9000, 11000, 13000];
    steps.forEach((step, i) => {
        setTimeout(() => {
            if (i > 0) { steps[i - 1].classList.remove('active'); steps[i - 1].classList.add('done'); }
            step.classList.add('active');
            document.getElementById('progressBar').style.width = `${((i + 1) / steps.length) * 100}%`;
        }, timings[i] || i * 2000);
    });
}

function completeProgress() {
    document.querySelectorAll('.p-step').forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
    document.getElementById('progressBar').style.width = '100%';
}

// ── Render Text-Guided Results ──────────────────────
function renderResults(data) {
    const section = document.getElementById('resultsSection');
    section.style.display = 'block';

    // Comparison images
    if (data.original_image) {
        document.getElementById('compBefore').src = 'data:image/jpeg;base64,' + data.original_image;
    }
    const afterSrc = data.final_overlay || getLastStepImage(data.step_images);
    if (afterSrc) {
        document.getElementById('compAfter').src = 'data:image/jpeg;base64,' + afterSrc;
    }

    // Reset slider
    document.getElementById('compOverlay').style.clipPath = 'inset(0 0 0 50%)';
    document.getElementById('compHandle').style.left = '50%';

    // Metrics
    document.getElementById('metricTime').textContent = data.time + 's';
    document.getElementById('metricParser').textContent = data.parsed?.parser_mode || 'rule-based';
    document.getElementById('metricSpatial').textContent = data.parsed?.spatial || 'none';
    document.getElementById('metricObject').textContent = data.parsed?.object_prompt || data.query;

    // Steps
    stepImages = data.step_images || {};
    const tabsContainer = document.getElementById('stepTabs');
    tabsContainer.innerHTML = '';
    const keys = Object.keys(stepImages);
    keys.forEach((name, i) => {
        const btn = document.createElement('button');
        btn.className = 'step-tab' + (i === 0 ? ' active' : '');
        btn.textContent = formatStepName(name);
        btn.onclick = () => showStep(name, btn);
        tabsContainer.appendChild(btn);
    });
    if (keys.length > 0) showStep(keys[0], tabsContainer.querySelector('.step-tab'));

    // Reasoning
    document.getElementById('reasoningText').textContent = data.reasoning || 'No reasoning output.';

    setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

function getLastStepImage(imgs) {
    if (!imgs) return null;
    const keys = Object.keys(imgs);
    return keys.length > 0 ? imgs[keys[keys.length - 1]] : null;
}

function formatStepName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
        .replace('Gdino', 'GDINO').replace('Clip', 'CLIP').replace('Sam', 'SAM');
}

function showStep(key, tab) {
    document.querySelectorAll('.step-tab').forEach(t => t.classList.remove('active'));
    if (tab) tab.classList.add('active');
    const img = document.getElementById('stepImage');
    img.src = stepImages[key] ? 'data:image/jpeg;base64,' + stepImages[key] : '';
}

// ── Render OOD Results ──────────────────────────────
function renderOodResults(data) {
    const section = document.getElementById('oodResults');
    section.style.display = 'block';

    // Metrics with animated bars
    if (data.metrics) {
        setOodMetric('oodIou', 'oodIouBar', data.metrics.iou);
        setOodMetric('oodF1', 'oodF1Bar', data.metrics.f1);
        setOodMetric('oodPrecision', 'oodPrecisionBar', data.metrics.precision);
        setOodMetric('oodRecall', 'oodRecallBar', data.metrics.recall);
    }

    // Result images
    const grid = document.getElementById('oodResultsGrid');
    grid.innerHTML = '';
    const imageTypes = [
        ['detection', 'Bounding Boxes'],
        ['masks', 'SAM Masks'],
        ['binary_mask', 'OOD Mask'],
    ];
    imageTypes.forEach(([key, label]) => {
        if (data.images && data.images[key]) {
            const card = document.createElement('div');
            card.className = 'ood-result-card';
            card.innerHTML = `
                <img src="data:image/jpeg;base64,${data.images[key]}" alt="${label}">
                <div class="ood-result-label">${label}</div>
            `;
            grid.appendChild(card);
        }
    });

    // Agent analysis
    const agentGrid = document.getElementById('agentCardsGrid');
    agentGrid.innerHTML = '';
    if (data.agents) {
        Object.entries(data.agents).forEach(([key, val]) => {
            const card = document.createElement('div');
            card.className = 'agent-card';
            const title = key.replace('agent', 'Agent ');
            const body = typeof val === 'object' ? JSON.stringify(val, null, 2) : String(val);
            card.innerHTML = `
                <div class="agent-card-header">${title}</div>
                <div class="agent-card-body"><pre style="white-space:pre-wrap;margin:0;font-size:11px">${body}</pre></div>
            `;
            agentGrid.appendChild(card);
        });
    }

    section.scrollIntoView({ behavior: 'smooth' });
}

function setOodMetric(valueId, barId, value) {
    const pct = value != null ? (value * 100).toFixed(1) : '--';
    document.getElementById(valueId).textContent = pct !== '--' ? pct + '%' : '--';
    setTimeout(() => {
        document.getElementById(barId).style.width = pct !== '--' ? pct + '%' : '0%';
    }, 200);
}

// ── Image Comparison Slider ─────────────────────────
function initComparison() {
    const container = document.getElementById('comparisonContainer');
    if (!container) return;
    let dragging = false;

    const update = (clientX) => {
        if (!dragging) return;
        const rect = container.getBoundingClientRect();
        let x = Math.max(0.03, Math.min(0.97, (clientX - rect.left) / rect.width));
        document.getElementById('compOverlay').style.clipPath = `inset(0 0 0 ${x * 100}%)`;
        document.getElementById('compHandle').style.left = (x * 100) + '%';
    };

    container.addEventListener('mousedown', (e) => { dragging = true; update(e.clientX); });
    document.addEventListener('mousemove', (e) => update(e.clientX));
    document.addEventListener('mouseup', () => dragging = false);
    container.addEventListener('touchstart', (e) => { dragging = true; update(e.touches[0].clientX); });
    document.addEventListener('touchmove', (e) => { if (dragging) update(e.touches[0].clientX); });
    document.addEventListener('touchend', () => dragging = false);
}
