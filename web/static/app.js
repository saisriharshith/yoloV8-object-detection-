/**
 * YOLOv8 Edge Video Analytics - Frontend Client Application
 * Clean, modern SPA for Real-Time Object Detection, Live Video Analytics, and Object Tracking.
 */

// Global State
const appState = {
    currentTab: 'camera',
    isStreaming: false,
    settings: {},
    systemMetrics: {},
    pollTimer: null
};

// Safe helper to set element text
function safeSetText(id, text) {
    const el = document.getElementById(id);
    if (el) el.innerText = text;
}

// DOM Content Loaded Initializer
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initTheme();
    loadSettings();
    loadSamples();
    initImageUpload();
    initVideoProcessing();
    initCameraControls();

    // Switch to camera tab by default
    switchTab('camera');

    // Start continuous telemetry polling (every 1 second)
    setInterval(updateLiveTelemetry, 1000);
});

// Navigation & Tab Switching
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTab = link.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });
}

function switchTab(tabId) {
    appState.currentTab = tabId;

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(l => {
        l.classList.toggle('active', l.getAttribute('data-tab') === tabId);
    });

    // Update tab panes
    document.querySelectorAll('.tab-pane').forEach(p => {
        p.classList.toggle('active', p.id === `tab-${tabId}`);
    });

    // Update Topbar Title
    const titleMap = {
        'camera': 'Live Camera Video Analytics',
        'video': 'Recorded Video Analysis & Export',
        'image': 'Static Image Object Detection',
        'settings': 'Inference Hyperparameters & Settings'
    };
    safeSetText('topbar-title', titleMap[tabId] || 'Real-Time YOLOv8 Analytics');
}

// Dark/Light Theme Switcher
function initTheme() {
    const themeBtn = document.getElementById('theme-toggle');
    if (!themeBtn) return;
    const savedTheme = localStorage.getItem('yolo_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeBtn.innerText = savedTheme === 'dark' ? '☀️ Light' : '🌙 Dark';

    themeBtn.addEventListener('click', () => {
        const cur = document.documentElement.getAttribute('data-theme');
        const next = cur === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('yolo_theme', next);
        themeBtn.innerText = next === 'dark' ? '☀️ Light' : '🌙 Dark';
    });
}

// Fetch & Populate Settings
async function loadSettings() {
    try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        appState.settings = data;

        // Model Select
        const modelSelect = document.getElementById('setting-model');
        if (modelSelect && data.available_models) {
            modelSelect.innerHTML = '';
            data.available_models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.innerText = m;
                if (m === data.model_name) opt.selected = true;
                modelSelect.appendChild(opt);
            });
        }

        // Device Select
        const devSelect = document.getElementById('setting-device');
        if (devSelect) devSelect.value = data.device;

        // Threshold Sliders
        const confSlider = document.getElementById('setting-conf');
        const confVal = document.getElementById('setting-conf-val');
        if (confSlider && confVal) {
            confSlider.value = data.conf_thresh;
            confVal.innerText = Number(data.conf_thresh).toFixed(2);
            confSlider.oninput = () => { confVal.innerText = Number(confSlider.value).toFixed(2); };
        }

        const iouSlider = document.getElementById('setting-iou');
        const iouVal = document.getElementById('setting-iou-val');
        if (iouSlider && iouVal) {
            iouSlider.value = data.iou_thresh;
            iouVal.innerText = Number(data.iou_thresh).toFixed(2);
            iouSlider.oninput = () => { iouVal.innerText = Number(iouSlider.value).toFixed(2); };
        }

        // Resolution
        const imgSizeSelect = document.getElementById('setting-imgsz');
        if (imgSizeSelect) imgSizeSelect.value = data.img_size;

        // Tracker
        const trackerSelect = document.getElementById('setting-tracker');
        if (trackerSelect) trackerSelect.value = data.tracker_type;

        // Class Filter Checkboxes
        const classGrid = document.getElementById('class-checkbox-grid');
        if (classGrid && data.all_classes) {
            classGrid.innerHTML = '';
            Object.entries(data.all_classes).forEach(([cid, cname]) => {
                const idNum = parseInt(cid);
                const isChecked = !data.enabled_classes || data.enabled_classes.includes(idNum);

                const item = document.createElement('label');
                item.style.display = 'inline-flex';
                item.style.alignItems = 'center';
                item.style.gap = '6px';
                item.style.fontSize = '12px';
                item.style.padding = '4px 8px';
                item.style.background = 'var(--bg-surface)';
                item.style.borderRadius = '4px';
                item.style.cursor = 'pointer';

                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.value = idNum;
                cb.checked = isChecked;
                cb.className = 'class-filter-cb';

                item.appendChild(cb);
                item.appendChild(document.createTextNode(cname));
                classGrid.appendChild(item);
            });
        }

        // Save Settings Button
        const saveBtn = document.getElementById('btn-save-settings');
        if (saveBtn) {
            saveBtn.onclick = async () => {
                saveBtn.disabled = true;
                saveBtn.innerText = 'Applying...';

                const selectedClasses = [];
                document.querySelectorAll('.class-filter-cb').forEach(cb => {
                    if (cb.checked) selectedClasses.push(parseInt(cb.value));
                });

                const payload = {
                    model_name: modelSelect ? modelSelect.value : undefined,
                    device: devSelect ? devSelect.value : undefined,
                    conf_thresh: confSlider ? parseFloat(confSlider.value) : undefined,
                    iou_thresh: iouSlider ? parseFloat(iouSlider.value) : undefined,
                    img_size: imgSizeSelect ? parseInt(imgSizeSelect.value) : undefined,
                    tracker_type: trackerSelect ? trackerSelect.value : undefined,
                    enabled_classes: selectedClasses.length > 0 ? selectedClasses : null
                };

                try {
                    const putRes = await fetch('/api/settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const putData = await putRes.json();
                    alert('Settings updated successfully!');
                } catch (e) {
                    alert('Failed to save settings: ' + e);
                } finally {
                    saveBtn.disabled = false;
                    saveBtn.innerText = 'Save & Apply Changes';
                }
            };
        }

    } catch (err) {
        console.error('Failed to load settings:', err);
    }
}

// Load Samples for Video Analysis
async function loadSamples() {
    try {
        const res = await fetch('/api/samples');
        const data = await res.json();

        const videoSelect = document.getElementById('sample-video-select');
        if (videoSelect && data.sample_videos) {
            videoSelect.innerHTML = '<option value="">-- Choose Sample Video --</option>';
            data.sample_videos.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v;
                opt.innerText = v;
                videoSelect.appendChild(opt);
            });
            // Auto-select sample_traffic.mp4 if present
            if (data.sample_videos.includes('sample_traffic.mp4')) {
                videoSelect.value = 'sample_traffic.mp4';
            }
        }
    } catch (err) {
        console.error('Failed to load samples:', err);
    }
}

// Live Camera & YouTube Live Controls
function initCameraControls() {
    const btnStartCamera = document.getElementById('btn-start-camera');
    const btnStopCamera = document.getElementById('btn-stop-camera');
    const btnSnapshot = document.getElementById('btn-snapshot');
    const streamImg = document.getElementById('live-stream-viewport');
    const inputTypeSelect = document.getElementById('camera-input-type');
    const ytContainer = document.getElementById('yt-input-container');
    const webcamContainer = document.getElementById('webcam-input-container');
    const ytUrlInput = document.getElementById('youtube-url-input');
    const btnPresetVenice = document.getElementById('btn-preset-venice');
    const cameraSourceInput = document.getElementById('camera-source-input');
    const cameraModeSelect = document.getElementById('camera-mode-select');

    if (inputTypeSelect) {
        inputTypeSelect.addEventListener('change', () => {
            if (inputTypeSelect.value === 'youtube') {
                if (ytContainer) ytContainer.style.display = 'inline-flex';
                if (webcamContainer) webcamContainer.style.display = 'none';
            } else {
                if (ytContainer) ytContainer.style.display = 'none';
                if (webcamContainer) webcamContainer.style.display = 'inline-flex';
            }
        });
    }

    if (btnPresetVenice) {
        btnPresetVenice.addEventListener('click', () => {
            if (ytUrlInput) {
                ytUrlInput.value = 'https://www.youtube.com/live/EO_1LWqsCNE?si=5U82EKHBjFXQRTYO';
            }
        });
    }

    if (btnStartCamera) {
        btnStartCamera.addEventListener('click', async () => {
            let src = '0';
            const isYT = inputTypeSelect && inputTypeSelect.value === 'youtube';
            if (isYT) {
                src = ytUrlInput ? ytUrlInput.value.trim() : '';
                if (!src) {
                    alert('Please enter a valid YouTube live stream URL.');
                    return;
                }
            } else {
                src = cameraSourceInput ? cameraSourceInput.value : '0';
            }

            const mode = cameraModeSelect ? cameraModeSelect.value : 'track';

            btnStartCamera.disabled = true;
            btnStartCamera.innerText = isYT ? 'Resolving Live Stream...' : 'Starting Camera...';

            try {
                const res = await fetch('/api/stream/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'start', source: src, mode: mode, record: false })
                });
                const data = await res.json();
                if (data.status === 'started') {
                    appState.isStreaming = true;
                    if (streamImg) streamImg.src = `/api/stream/video_feed?t=${Date.now()}`;
                    const dot = document.getElementById('camera-status-dot');
                    if (dot) dot.style.display = 'inline-flex';
                    btnStartCamera.disabled = true;
                    btnStartCamera.innerText = 'Stream Running';
                    if (btnStopCamera) btnStopCamera.disabled = false;
                }
            } catch (err) {
                alert('Failed to start stream: ' + err);
                btnStartCamera.disabled = false;
                btnStartCamera.innerText = '▶ Start Stream';
            }
        });
    }

    if (btnStopCamera) {
        btnStopCamera.addEventListener('click', async () => {
            try {
                btnStopCamera.disabled = true;
                await fetch('/api/stream/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'stop' })
                });
                appState.isStreaming = false;
                if (streamImg) streamImg.src = '';
                const dot = document.getElementById('camera-status-dot');
                if (dot) dot.style.display = 'none';
                if (btnStartCamera) {
                    btnStartCamera.disabled = false;
                    btnStartCamera.innerText = 'Start Camera';
                }
                btnStopCamera.disabled = true;
            } catch (err) {
                console.error('Failed to stop camera:', err);
            }
        });
    }

    if (btnSnapshot) {
        btnSnapshot.addEventListener('click', () => {
            if (!appState.isStreaming || !streamImg || !streamImg.src) {
                alert('Camera is not active.');
                return;
            }
            const a = document.createElement('a');
            a.href = streamImg.src;
            a.download = `snapshot_${Date.now()}.jpg`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        });
    }
}

// Update Live Telemetry
async function updateLiveTelemetry() {
    if (!appState.isStreaming) return;
    try {
        const res = await fetch('/api/stream/status');
        const data = await res.json();
        const m = data.metrics;
        if (!m) return;

        // Display stream title if active
        if (data.stream_title) {
            safeSetText('camera-viewport-title', `🔴 ${data.stream_title}`);
        } else if (data.is_streaming) {
            safeSetText('camera-viewport-title', 'FaceTime HD / Live Stream Viewport');
        }

        // Pipeline speed and latency
        safeSetText('live-fps-val', Number(m.fps).toFixed(1));
        safeSetText('live-lat-val', `${Number(m.latency_ms).toFixed(1)} ms`);
        safeSetText('live-cur-val', m.current_count);
        safeSetText('live-uniq-val', m.cumulative_unique_total);

        // Friendly category counters: Cars, Persons, Bikes, Trucks
        if (m.categories) {
            safeSetText('live-cars', m.categories.cars || 0);
            safeSetText('live-persons', m.categories.persons || 0);
            safeSetText('live-bikes', m.categories.bikes || 0);
            safeSetText('live-trucks', m.categories.trucks || 0);

            safeSetText('video-cars', m.categories.cars || 0);
            safeSetText('video-persons', m.categories.persons || 0);
            safeSetText('video-bikes', m.categories.bikes || 0);
            safeSetText('video-trucks', m.categories.trucks || 0);
        }

        // Video Analysis tab live telemetry
        safeSetText('video-live-fps', Number(m.fps).toFixed(1));
        safeSetText('video-live-cur', m.current_count);
        safeSetText('video-live-uniq', m.cumulative_unique_total);
        if (data.total_frames > 0) {
            safeSetText('video-frame-progress', `${data.frame_idx} / ${data.total_frames}`);
            const pct = Math.min(100, Math.round((data.frame_idx / data.total_frames) * 100));
            safeSetText('video-frame-pct', `${pct}% Analyzed`);
        }

        // Class breakdown chips for live camera
        const camClassContainer = document.getElementById('camera-classes-summary');
        if (camClassContainer && m.current_classes) {
            camClassContainer.innerHTML = '';
            for (const [cls, count] of Object.entries(m.current_classes)) {
                const chip = document.createElement('span');
                chip.style.padding = '3px 8px';
                chip.style.background = 'var(--bg-elevated)';
                chip.style.borderRadius = '12px';
                chip.style.fontSize = '11px';
                chip.style.border = '1px solid var(--border-color)';
                chip.innerText = `${cls}: ${count}`;
                camClassContainer.appendChild(chip);
            }
        }

        // Class breakdown chips for video analysis
        const vidClassContainer = document.getElementById('video-classes-summary');
        if (vidClassContainer && m.current_classes) {
            vidClassContainer.innerHTML = '';
            for (const [cls, count] of Object.entries(m.current_classes)) {
                const chip = document.createElement('span');
                chip.style.padding = '3px 8px';
                chip.style.background = 'var(--bg-elevated)';
                chip.style.borderRadius = '12px';
                chip.style.fontSize = '11px';
                chip.style.border = '1px solid var(--border-color)';
                chip.innerText = `${cls}: ${count}`;
                vidClassContainer.appendChild(chip);
            }
        }

        // Automatic completion check for recorded video analysis
        if (data.export_completed) {
            const resultsBox = document.getElementById('video-results-box');
            if (resultsBox) {
                resultsBox.style.display = 'block';
                const dlMp4 = document.getElementById('btn-download-mp4');
                const dlCsv = document.getElementById('btn-download-csv');
                if (dlMp4 && data.video_download_url) dlMp4.href = data.video_download_url;
                if (dlCsv && data.csv_download_url) dlCsv.href = data.csv_download_url;
            }
            const btnStart = document.getElementById('btn-start-video-live');
            const btnStop = document.getElementById('btn-stop-video-live');
            if (btnStart) {
                btnStart.disabled = false;
                btnStart.innerText = '▶ Play & Analyze Live';
            }
            if (btnStop) btnStop.disabled = true;
        }

    } catch (err) {
        // quiet catch
    }
}

// Static Image Upload & Inference
function initImageUpload() {
    const fileInput = document.getElementById('img-file-input');
    const runBtn = document.getElementById('btn-run-image-detect');
    const previewOrig = document.getElementById('img-preview-orig');
    const origPlaceholder = document.getElementById('img-orig-placeholder');
    const previewAnnotated = document.getElementById('img-preview-annotated');
    const annotPlaceholder = document.getElementById('img-annot-placeholder');
    const tableBody = document.querySelector('#image-detections-table tbody');

    if (!fileInput || !runBtn) return;

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            if (previewOrig) {
                previewOrig.src = URL.createObjectURL(file);
                previewOrig.style.display = 'block';
            }
            if (origPlaceholder) origPlaceholder.style.display = 'none';
            runBtn.disabled = false;
        }
    });

    runBtn.addEventListener('click', async () => {
        if (!fileInput.files.length) return;
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        runBtn.disabled = true;
        runBtn.innerText = 'Detecting...';

        try {
            const res = await fetch('/api/detect/image', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            // Display annotated preview
            if (previewAnnotated) {
                previewAnnotated.src = data.annotated_image_base64;
                previewAnnotated.style.display = 'block';
            }
            if (annotPlaceholder) annotPlaceholder.style.display = 'none';

            safeSetText('img-kpi-count', data.objects_count);
            safeSetText('img-kpi-latency', `${data.latency_ms} ms`);
            safeSetText('img-kpi-fps', data.latency_ms > 0 ? (1000 / data.latency_ms).toFixed(1) : 'N/A');

            // Populate coordinates table
            if (tableBody) {
                tableBody.innerHTML = '';
                if (!data.detections || data.detections.length === 0) {
                    tableBody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No objects detected at current threshold.</td></tr>';
                } else {
                    data.detections.forEach((d, i) => {
                        const row = document.createElement('tr');
                        const [x1, y1, x2, y2] = d.box.map(Math.round);
                        row.innerHTML = `
                            <td>${i + 1}</td>
                            <td><strong style="color:var(--accent-cyan);">${d.class_name}</strong></td>
                            <td>${(d.confidence * 100).toFixed(1)}%</td>
                            <td style="font-family:var(--font-mono); font-size:12px;">[${x1}, ${y1}, ${x2}, ${y2}]</td>
                            <td>${x2 - x1} x ${y2 - y1} px</td>
                        `;
                        tableBody.appendChild(row);
                    });
                }
            }
        } catch (err) {
            alert('Image detection failed: ' + err);
        } finally {
            runBtn.disabled = false;
            runBtn.innerText = 'Run Detection';
        }
    });
}

// Video Processing & Live Stream Analysis
function initVideoProcessing() {
    const btnStart = document.getElementById('btn-start-video-live');
    const btnStop = document.getElementById('btn-stop-video-live');
    const videoViewport = document.getElementById('video-analysis-viewport');
    const resultsBox = document.getElementById('video-results-box');
    const fileInput = document.getElementById('video-file-input');
    const sampleSelect = document.getElementById('sample-video-select');
    const videoYtInput = document.getElementById('video-yt-input');
    const modeSelect = document.getElementById('video-mode-select');

    if (!btnStart) return;

    btnStart.addEventListener('click', async () => {
        let videoSource = '';
        if (videoYtInput && videoYtInput.value.trim()) {
            videoSource = videoYtInput.value.trim();
        } else if (fileInput && fileInput.files.length > 0) {
            // Upload custom video first
            const uploadData = new FormData();
            uploadData.append('file', fileInput.files[0]);
            btnStart.disabled = true;
            btnStart.innerText = 'Uploading video...';
            try {
                const upRes = await fetch('/api/upload/video', { method: 'POST', body: uploadData });
                const upData = await upRes.json();
                videoSource = upData.filename;
            } catch (e) {
                alert('Upload failed: ' + e);
                btnStart.disabled = false;
                btnStart.innerText = '▶ Play & Analyze Live';
                return;
            }
        } else if (sampleSelect && sampleSelect.value) {
            videoSource = sampleSelect.value;
        } else {
            alert('Please choose a sample video or upload a video file.');
            return;
        }

        const mode = modeSelect?.value || 'track';
        btnStart.disabled = true;
        btnStart.innerText = 'Analyzing Video Live...';
        if (resultsBox) resultsBox.style.display = 'none';

        try {
            const res = await fetch('/api/stream/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'start', source: videoSource, mode: mode, record: true })
            });
            const data = await res.json();
            if (data.status === 'started') {
                appState.isStreaming = true;
                if (videoViewport) videoViewport.src = `/api/stream/video_feed?t=${Date.now()}`;
                if (btnStop) btnStop.disabled = false;
            }
        } catch (err) {
            alert('Failed to start video analysis: ' + err);
            btnStart.disabled = false;
            btnStart.innerText = '▶ Play & Analyze Live';
        }
    });

    if (btnStop) {
        btnStop.addEventListener('click', async () => {
            try {
                btnStop.disabled = true;
                const res = await fetch('/api/stream/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'stop' })
                });
                const data = await res.json();
                appState.isStreaming = false;
                if (videoViewport) videoViewport.src = '';
                btnStart.disabled = false;
                btnStart.innerText = '▶ Play & Analyze Live';

                if (data.export_completed && resultsBox) {
                    resultsBox.style.display = 'block';
                    const dlMp4 = document.getElementById('btn-download-mp4');
                    const dlCsv = document.getElementById('btn-download-csv');
                    if (dlMp4) dlMp4.href = data.video_download_url;
                    if (dlCsv) dlCsv.href = data.csv_download_url;
                }
            } catch (err) {
                console.error('Stop error:', err);
            }
        });
    }
}
