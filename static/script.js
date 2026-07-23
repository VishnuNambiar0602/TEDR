document.addEventListener('DOMContentLoaded', () => {
    const uploadTrigger = document.getElementById('upload-trigger');
    const fileInput = document.getElementById('file-input');
    const uploadTitle = document.getElementById('upload-title');
    const uploadHint = document.getElementById('upload-hint');
    const loadingText = document.getElementById('loading-text');
    const uploadSection = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');
    const modeButtons = document.querySelectorAll('.mode-btn');
    const videoOptions = document.getElementById('video-options');
    const frameSkipInput = document.getElementById('frame-skip');
    const resultImage = document.getElementById('result-image');
    const resultVideo = document.getElementById('result-video');
    const resultCanvas = document.getElementById('result-canvas');
    const downloadVideoBtn = document.getElementById('download-video-btn');
    const downloadReportBtn = document.getElementById('download-report-btn');
    const framesStat = document.getElementById('frames-stat');
    const analyzedFrames = document.getElementById('analyzed-frames');
    const vehicleCount = document.getElementById('vehicle-count');
    const occupancyRatio = document.getElementById('occupancy-ratio');
    const congestionLevel = document.getElementById('congestion-level');
    const congestionBadge = document.getElementById('congestion-badge');
    const resetBtn = document.getElementById('reset-btn');
    const processingMode = document.getElementById('processing-mode');
    const frameSkipContainer = document.getElementById('frame-skip-container');
    let selectedMode = 'image';

    processingMode.addEventListener('change', () => {
        if (processingMode.value === 'live') {
            frameSkipContainer.classList.remove('hidden');
        } else {
            frameSkipContainer.classList.add('hidden');
        }
    });

    modeButtons.forEach((button) => {
        button.addEventListener('click', () => {
            selectedMode = button.dataset.mode;
            modeButtons.forEach((b) => b.classList.remove('active'));
            button.classList.add('active');
            updateModeUI();
        });
    });

    function updateModeUI() {
        if (selectedMode === 'video') {
            uploadTitle.innerText = 'Upload Traffic Video';
            uploadHint.innerText = 'Drag & drop or click to browse .mp4, .avi, .mov, .mkv';
            fileInput.setAttribute('accept', 'video/*');
            videoOptions.classList.remove('hidden');
        } else {
            uploadTitle.innerText = 'Upload Traffic Image';
            uploadHint.innerText = 'Drag & drop or click to browse';
            fileInput.setAttribute('accept', 'image/*');
            videoOptions.classList.add('hidden');
        }
    }

    // Handle Click
    uploadTrigger.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle File Selection
    fileInput.addEventListener('change', handleFile);

    // Handle Drag & Drop
    const uploadCard = document.querySelector('.upload-card');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadCard.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadCard.style.borderColor = '#3b82f6';
        uploadCard.style.background = 'rgba(255, 255, 255, 0.1)';
    }

    function unhighlight(e) {
        uploadCard.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        uploadCard.style.background = 'rgba(255, 255, 255, 0.05)';
    }

    uploadCard.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFiles(files);
        }
    }

    function handleFile(e) {
        if (e.target.files.length) {
            handleFiles(e.target.files);
        }
    }

    function handleFiles(files) {
        const file = files[0];
        if (selectedMode === 'image') {
            if (file.type.startsWith('image/')) {
                uploadAndAnalyzeImage(file);
            } else {
                alert('Please upload an image file.');
            }
            return;
        }

        if (file.type.startsWith('video/')) {
            playAndAnalyzeVideoLocally(file);
        } else {
            alert('Please upload a video file.');
        }
    }

    function uploadAndAnalyzeImage(file) {
        // Show loading
        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        loadingText.innerText = 'Analyzing Traffic Image...';

        const formData = new FormData();
        formData.append('file', file);

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showImageResults(data);
                } else {
                    alert('Error analyzing image: ' + data.error);
                    resetView();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during upload.');
                resetView();
            });
    }

    function uploadAndAnalyzeVideo(file) {
        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        loadingText.innerText = 'Uploading Traffic Video...';

        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload_video', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    startRealTimeProcessing(data.video_id, data.total_frames);
                } else {
                    alert('Error uploading video: ' + data.error);
                    resetView();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during upload.');
                resetView();
            });
    }

    function startRealTimeProcessing(videoId, totalFrames) {
        loadingText.innerText = 'Initializing Real-Time Object Detection...';
        
        // Show the results card immediately, so the user can watch the stream!
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
        resultVideo.classList.add('hidden');
        resultCanvas.classList.add('hidden'); // Hide local canvas overlay
        resultImage.classList.remove('hidden'); // Show image tag for streaming frames!
        framesStat.classList.remove('hidden');
        downloadVideoBtn.classList.add('hidden'); // Hide download button until complete
        
        let frameSkipValue = 1;
        if (processingMode.value === 'live') {
            frameSkipValue = Math.max(1, parseInt(frameSkipInput.value || '2', 10));
        }
        
        // Setup WebSocket connection
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/process_video/${videoId}?frame_skip=${frameSkipValue}`;
        const ws = new WebSocket(wsUrl);
        
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            
            if (msg.type === 'frame') {
                // Update image with the latest frame
                resultImage.src = msg.frame;
                
                // Update statistics in real-time
                vehicleCount.innerText = msg.vehicle_count;
                occupancyRatio.innerText = (msg.occupancy_ratio * 100).toFixed(1) + '%';
                congestionLevel.innerText = msg.congestion_level;
                
                // Update badge
                congestionBadge.innerText = msg.congestion_level;
                congestionBadge.className = 'badge ' + msg.congestion_level.toLowerCase();
                
                // Update frame count progress
                analyzedFrames.innerText = `${msg.frame_index} / ${msg.total_frames}`;
            } else if (msg.type === 'complete') {
                // Update final averages
                vehicleCount.innerText = msg.average_vehicle_count;
                occupancyRatio.innerText = (msg.average_occupancy_ratio * 100).toFixed(1) + '%';
                congestionLevel.innerText = msg.congestion_level;
                
                congestionBadge.innerText = msg.congestion_level;
                congestionBadge.className = 'badge ' + msg.congestion_level.toLowerCase();
                
                analyzedFrames.innerText = `${msg.analyzed_frames} / ${msg.total_frames}`;
                
                // Show download button
                downloadVideoBtn.href = msg.download_url;
                downloadVideoBtn.classList.remove('hidden');
                
                downloadReportBtn.href = generateReport(msg, 'video');
                downloadReportBtn.classList.remove('hidden');
                
                ws.close();
            } else if (msg.type === 'error') {
                alert('Error processing video: ' + msg.message);
                resetView();
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            alert('Connection lost during processing.');
            resetView();
        };
    }

    function generateReport(data, type) {
        let tableRows = '';
        if (type === 'image') {
            tableRows = `
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Vehicle Count</td><td>${data.vehicle_count}</td></tr>
                <tr><td>Occupancy Ratio</td><td>${(data.occupancy_ratio * 100).toFixed(1)}%</td></tr>
                <tr><td>Congestion Level</td><td>${data.congestion_level}</td></tr>
            `;
        } else {
            tableRows = `
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Average Vehicle Count</td><td>${data.average_vehicle_count}</td></tr>
                <tr><td>Average Occupancy Ratio</td><td>${(data.average_occupancy_ratio * 100).toFixed(1)}%</td></tr>
                <tr><td>Dominant Congestion Level</td><td>${data.congestion_level || data.dominant_congestion}</td></tr>
                <tr><td>Analyzed Frames</td><td>${data.analyzed_frames} / ${data.total_frames}</td></tr>
            `;
        }

        const htmlContent = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Traffic Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 40px; margin: 0; }
                    h1 { text-align: center; color: #333; }
                    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <h1>Traffic Analysis Report</h1>
                <p>Generated on: ${new Date().toLocaleString()}</p>
                <table>
                    ${tableRows}
                </table>
            </body>
            </html>
        `;

        const blob = new Blob([htmlContent], { type: 'text/html' });
        return URL.createObjectURL(blob);
    }

    function showImageResults(data) {
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
        downloadVideoBtn.classList.add('hidden');
        framesStat.classList.add('hidden');
        resultVideo.classList.add('hidden');
        resultImage.classList.remove('hidden');

        // Update UI
        resultImage.src = data.image;
        vehicleCount.innerText = data.vehicle_count;
        occupancyRatio.innerText = (data.occupancy_ratio * 100).toFixed(1) + '%';
        congestionLevel.innerText = data.congestion_level;

        // Badge styling
        congestionBadge.innerText = data.congestion_level;
        congestionBadge.className = 'badge ' + data.congestion_level.toLowerCase();
        
        downloadReportBtn.href = generateReport(data, 'image');
        downloadReportBtn.classList.remove('hidden');
    }

    function getDominantCongestion(congestionCounts) {
        if (!congestionCounts) {
            return 'LOW';
        }

        let dominant = 'LOW';
        let maxCount = -1;
        ['LOW', 'MEDIUM', 'HIGH'].forEach((level) => {
            const count = congestionCounts[level] || 0;
            if (count > maxCount) {
                maxCount = count;
                dominant = level;
            }
        });
        return dominant;
    }

    function showVideoResults(data) {
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
        resultImage.classList.add('hidden');
        resultVideo.classList.remove('hidden');
        framesStat.classList.remove('hidden');

        const dominantCongestion = getDominantCongestion(data.congestion_frame_counts);
        const videoUrl = data.download_url + '?t=' + Date.now();

        resultVideo.src = videoUrl;
        resultVideo.load();
        downloadVideoBtn.href = data.download_url;
        downloadVideoBtn.classList.remove('hidden');

        vehicleCount.innerText = data.average_vehicle_count;
        occupancyRatio.innerText = (data.average_occupancy_ratio * 100).toFixed(1) + '%';
        congestionLevel.innerText = dominantCongestion;
        analyzedFrames.innerText = `${data.analyzed_frames} / ${data.total_frames}`;

        congestionBadge.innerText = dominantCongestion;
        congestionBadge.className = 'badge ' + dominantCongestion.toLowerCase();
        
        data.congestion_level = dominantCongestion;
        downloadReportBtn.href = generateReport(data, 'video');
        downloadReportBtn.classList.remove('hidden');
    }
    let wsDetect = null;
    let isWaitingForResponse = false;
    let animationFrameId = null;
    let isSyncLoopActive = false;
    let measuredFPS = 25;
    let measuredFrameDuration = 0.04;
    let isMeasuringFPS = false;
    let rVFCId = null;
    let sendTime = 0;
    let averageInferenceTime = 500;
    let isPausedForWaiting = false;

    function playAndAnalyzeVideoLocally(file) {
        uploadSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
        resultImage.classList.add('hidden');
        resultVideo.classList.remove('hidden');
        resultCanvas.classList.remove('hidden');
        framesStat.classList.add('hidden');
        downloadVideoBtn.classList.add('hidden');

        stopLocalAnalysis();

        const localUrl = URL.createObjectURL(file);
        resultVideo.src = localUrl;
        resultVideo.muted = true;
        resultVideo.loop = true;
        resultVideo.load();
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsDetect = new WebSocket(`${protocol}//${window.location.host}/ws/detect_frame`);
        
        let wsReady = false;
        let fpsReady = false;

        function startProcessingIfReady() {
            if (wsReady && fpsReady) {
                console.log("WebSocket and FPS estimation ready. Starting analysis loop.");
                startAnalysisLoop();
            }
        }

        wsDetect.onopen = () => {
            console.log("WebSocket connected for real-time frame detection");
            wsReady = true;
            startProcessingIfReady();
        };

        wsDetect.onmessage = (event) => {
            isWaitingForResponse = false;
            const data = JSON.parse(event.data);
            if (data.error) {
                console.error("Server error:", data.error);
                return;
            }

            drawDetections(data);

            if (processingMode.value === 'sync') {
                const latency = Date.now() - sendTime;
                averageInferenceTime = averageInferenceTime * 0.8 + latency * 0.2;
                
                const targetFPS = 1000 / averageInferenceTime;
                const targetPlaybackRate = Math.min(1.0, targetFPS / measuredFPS);
                resultVideo.playbackRate = Math.max(0.05, targetPlaybackRate);

                if (isPausedForWaiting) {
                    resultVideo.play().then(() => {
                        isPausedForWaiting = false;
                    }).catch(e => {});
                }
            }
        };

        wsDetect.onerror = (err) => {
            console.error("WebSocket error:", err);
        };

        wsDetect.onclose = () => {
            console.log("WebSocket closed");
            stopLocalAnalysis();
        };

        // Measure video FPS when video starts loading
        resultVideo.onloadedmetadata = () => {
            console.log("Video metadata loaded. Measuring FPS...");
            isMeasuringFPS = true;
            let frameTimes = [];
            
            // Play video to trigger frames
            resultVideo.play().then(() => {
                function measureFPSCallback(now, metadata) {
                    if (!isMeasuringFPS) return;
                    if (metadata.mediaTime !== undefined) {
                        frameTimes.push(metadata.mediaTime);
                        if (frameTimes.length >= 10) {
                            let diffs = [];
                            for (let i = 1; i < frameTimes.length; i++) {
                                let diff = frameTimes[i] - frameTimes[i-1];
                                if (diff > 0 && diff < 0.2) {
                                    diffs.push(diff);
                                }
                            }
                            if (diffs.length > 0) {
                                measuredFrameDuration = diffs.reduce((a, b) => a + b, 0) / diffs.length;
                                measuredFPS = Math.round(1 / measuredFrameDuration);
                                console.log(`Measured video FPS: ${measuredFPS}, frame duration: ${measuredFrameDuration}`);
                                
                                isMeasuringFPS = false;
                                fpsReady = true;
                                
                                if (processingMode.value === 'sync') {
                                    resultVideo.pause();
                                }
                                
                                startProcessingIfReady();
                                return;
                            }
                        }
                    }
                    if (resultVideo.requestVideoFrameCallback) {
                        rVFCId = resultVideo.requestVideoFrameCallback(measureFPSCallback);
                    }
                }

                if (resultVideo.requestVideoFrameCallback) {
                    rVFCId = resultVideo.requestVideoFrameCallback(measureFPSCallback);
                } else {
                    // Fallback if requestVideoFrameCallback is not supported
                    setTimeout(() => {
                        console.log("requestVideoFrameCallback not supported or delayed. Using default FPS (25).");
                        isMeasuringFPS = false;
                        measuredFPS = 25;
                        measuredFrameDuration = 0.04;
                        fpsReady = true;
                        if (processingMode.value === 'sync') {
                            resultVideo.pause();
                        }
                        startProcessingIfReady();
                    }, 500);
                }
            }).catch(err => {
                console.error("Error playing video for FPS measurement, using fallback:", err);
                isMeasuringFPS = false;
                measuredFPS = 25;
                measuredFrameDuration = 0.04;
                fpsReady = true;
                if (processingMode.value === 'sync') {
                    resultVideo.pause();
                }
                startProcessingIfReady();
            });
        };
    }

    function stopLocalAnalysis() {
        isSyncLoopActive = false;
        resultVideo.onplay = null;
        resultVideo.onpause = null;
        resultVideo.onseeked = null;
        resultVideo.onloadedmetadata = null;
        if (wsDetect) {
            try { wsDetect.close(); } catch(e) {}
            wsDetect = null;
        }
        isWaitingForResponse = false;
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (rVFCId) {
            if (resultVideo.cancelVideoFrameCallback) {
                resultVideo.cancelVideoFrameCallback(rVFCId);
            } else {
                cancelAnimationFrame(rVFCId);
            }
            rVFCId = null;
        }
        isMeasuringFPS = false;
        isPausedForWaiting = false;
        sendTime = 0;
        averageInferenceTime = 500;
        
        lastRectWidth = 0;
        lastRectHeight = 0;
        activeTracks = [];
        nextTrackId = 1;

        const ctx = resultCanvas.getContext('2d');
        ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    }

    const offscreenCanvas = document.createElement('canvas');
    const offscreenCtx = offscreenCanvas.getContext('2d');

    function sendSyncFrame() {
        if (!isSyncLoopActive || !wsDetect || wsDetect.readyState !== WebSocket.OPEN) return;
        if (isWaitingForResponse) return;

        const W = resultVideo.videoWidth;
        const H = resultVideo.videoHeight;
        if (!W || !H) {
            setTimeout(sendSyncFrame, 50);
            return;
        }

        const capWidth = 640;
        const capHeight = Math.round((H / W) * capWidth);
        offscreenCanvas.width = capWidth;
        offscreenCanvas.height = capHeight;
        
        offscreenCtx.drawImage(resultVideo, 0, 0, capWidth, capHeight);
        
        isWaitingForResponse = true;
        sendTime = Date.now();
        offscreenCanvas.toBlob((blob) => {
            if (blob && wsDetect && wsDetect.readyState === WebSocket.OPEN) {
                wsDetect.send(blob);
            } else {
                isWaitingForResponse = false;
            }
        }, 'image/jpeg', 0.6);
    }

    function startAnalysisLoop() {
        const mode = processingMode.value;

        if (mode === 'live') {
            resultVideo.playbackRate = 1.0;
            if (resultVideo.paused) {
                resultVideo.play().catch(e => console.error("Error playing video:", e));
            }

            function liveFrameCallback(now, metadata) {
                if (processingMode.value !== 'live' || !wsDetect || wsDetect.readyState !== WebSocket.OPEN) return;

                if (!resultVideo.paused && !resultVideo.ended && !isWaitingForResponse) {
                    const W = resultVideo.videoWidth;
                    const H = resultVideo.videoHeight;
                    if (W && H) {
                        const capWidth = 640;
                        const capHeight = Math.round((H / W) * capWidth);
                        offscreenCanvas.width = capWidth;
                        offscreenCanvas.height = capHeight;
                        
                        offscreenCtx.drawImage(resultVideo, 0, 0, capWidth, capHeight);
                        
                        isWaitingForResponse = true;
                        offscreenCanvas.toBlob((blob) => {
                            if (blob && wsDetect && wsDetect.readyState === WebSocket.OPEN) {
                                wsDetect.send(blob);
                            } else {
                                isWaitingForResponse = false;
                            }
                        }, 'image/jpeg', 0.6);
                    }
                }

                if (resultVideo.requestVideoFrameCallback) {
                    rVFCId = resultVideo.requestVideoFrameCallback(liveFrameCallback);
                } else {
                    rVFCId = requestAnimationFrame(() => {
                        liveFrameCallback(performance.now(), {});
                    });
                }
            }

            if (resultVideo.requestVideoFrameCallback) {
                rVFCId = resultVideo.requestVideoFrameCallback(liveFrameCallback);
            } else {
                rVFCId = requestAnimationFrame(() => {
                    liveFrameCallback(performance.now(), {});
                });
            }
        } else {
            // Sync Mode (advance frame-by-frame as processed)
            isSyncLoopActive = true;
            isPausedForWaiting = false;

            resultVideo.playbackRate = 0.05; // Initial slow rate
            if (resultVideo.paused) {
                resultVideo.play().catch(e => console.error("Error playing video:", e));
            }

            function syncFrameCallback(now, metadata) {
                if (processingMode.value !== 'sync' || !isSyncLoopActive || !wsDetect || wsDetect.readyState !== WebSocket.OPEN) return;

                if (isWaitingForResponse) {
                    if (!resultVideo.paused) {
                        resultVideo.pause();
                        isPausedForWaiting = true;
                    }
                    rVFCId = resultVideo.requestVideoFrameCallback(syncFrameCallback);
                    return;
                }

                const W = resultVideo.videoWidth;
                const H = resultVideo.videoHeight;
                if (W && H) {
                    const capWidth = 640;
                    const capHeight = Math.round((H / W) * capWidth);
                    offscreenCanvas.width = capWidth;
                    offscreenCanvas.height = capHeight;
                    
                    offscreenCtx.drawImage(resultVideo, 0, 0, capWidth, capHeight);
                    
                    isWaitingForResponse = true;
                    sendTime = Date.now();
                    offscreenCanvas.toBlob((blob) => {
                        if (blob && wsDetect && wsDetect.readyState === WebSocket.OPEN) {
                            wsDetect.send(blob);
                        } else {
                            isWaitingForResponse = false;
                        }
                    }, 'image/jpeg', 0.6);
                }

                rVFCId = resultVideo.requestVideoFrameCallback(syncFrameCallback);
            }

            if (resultVideo.requestVideoFrameCallback) {
                rVFCId = resultVideo.requestVideoFrameCallback(syncFrameCallback);
            } else {
                rVFCId = requestAnimationFrame(() => {
                    syncFrameCallback(performance.now(), {});
                });
            }
        }
        
        lastAnimTime = Date.now();
        function updateCanvasSize() {
            if (resultVideo.paused && mode === 'live') {
                animationFrameId = requestAnimationFrame(updateCanvasSize);
                return;
            }
            
            resizeCanvasToVideo();
            drawActiveTracks();
            animationFrameId = requestAnimationFrame(updateCanvasSize);
        }
        animationFrameId = requestAnimationFrame(updateCanvasSize);
    }

    let lastRectWidth = 0;
    let lastRectHeight = 0;
    let activeTracks = [];
    let nextTrackId = 1;
    let lastAnimTime = Date.now();

    function getIoU(box1, box2) {
        const xA = Math.max(box1[0], box2[0]);
        const yA = Math.max(box1[1], box2[1]);
        const xB = Math.min(box1[2], box2[2]);
        const yB = Math.min(box1[3], box2[3]);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        if (interArea === 0) return 0;

        const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        return interArea / (box1Area + box2Area - interArea);
    }

    function getDistance(box1, box2) {
        const c1x = (box1[0] + box1[2]) / 2;
        const c1y = (box1[1] + box1[3]) / 2;
        const c2x = (box2[0] + box2[2]) / 2;
        const c2y = (box2[1] + box2[3]) / 2;
        return Math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2);
    }

    function resizeCanvasToVideo() {
        const videoWidth = resultVideo.videoWidth;
        const videoHeight = resultVideo.videoHeight;
        if (!videoWidth || !videoHeight) return;
        
        const rect = resultVideo.getBoundingClientRect();
        
        // Only reset canvas dimensions when actual dimensions change, preventing constant resets/clears.
        if (Math.abs(rect.width - lastRectWidth) > 1 || Math.abs(rect.height - lastRectHeight) > 1) {
            lastRectWidth = rect.width;
            lastRectHeight = rect.height;
            
            resultCanvas.style.width = `${rect.width}px`;
            resultCanvas.style.height = `${rect.height}px`;
            resultCanvas.style.left = `${resultVideo.offsetLeft}px`;
            resultCanvas.style.top = `${resultVideo.offsetTop}px`;
            
            resultCanvas.width = rect.width;
            resultCanvas.height = rect.height;
        }
    }

    function drawDetections(data) {
        vehicleCount.innerText = data.vehicle_count;
        occupancyRatio.innerText = (data.occupancy_ratio * 100).toFixed(1) + '%';
        congestionLevel.innerText = data.congestion_level;
        
        congestionBadge.innerText = data.congestion_level;
        congestionBadge.className = 'badge ' + data.congestion_level.toLowerCase();

        const detections = data.detections;
        const serverW = data.width;
        const serverH = data.height;
        if (!serverW || !serverH) return;

        const scaleX = resultCanvas.width / serverW;
        const scaleY = resultCanvas.height / serverH;

        const now = Date.now();

        // 1. Scale new detections
        const newDets = detections.map(det => {
            return {
                class: det.class,
                confidence: det.confidence,
                bbox: [
                    det.bbox[0] * scaleX,
                    det.bbox[1] * scaleY,
                    det.bbox[2] * scaleX,
                    det.bbox[3] * scaleY
                ]
            };
        });

        // 2. Match existing tracks with new detections
        const matchedDets = new Set();
        
        activeTracks.forEach(track => {
            let bestMatchIdx = -1;
            let bestMatchVal = 0.15; // Min IoU threshold
            
            newDets.forEach((det, idx) => {
                if (matchedDets.has(idx)) return;
                if (det.class !== track.class) return;
                
                const iou = getIoU(track.targetBox, det.bbox);
                if (iou > bestMatchVal) {
                    bestMatchVal = iou;
                    bestMatchIdx = idx;
                }
            });
            
            // If no IoU match, try distance match for close objects
            if (bestMatchIdx === -1) {
                let minDistance = 80; // Max distance threshold in px
                newDets.forEach((det, idx) => {
                    if (matchedDets.has(idx)) return;
                    if (det.class !== track.class) return;
                    
                    const dist = getDistance(track.targetBox, det.bbox);
                    if (dist < minDistance) {
                        minDistance = dist;
                        bestMatchIdx = idx;
                    }
                });
            }
            
            if (bestMatchIdx !== -1) {
                const det = newDets[bestMatchIdx];
                matchedDets.add(bestMatchIdx);
                
                const dt = now - track.lastSeen;
                if (dt > 0) {
                    // Estimate velocity: change in position per millisecond
                    track.velocity = [
                        (det.bbox[0] - track.currentBox[0]) / dt,
                        (det.bbox[1] - track.currentBox[1]) / dt,
                        (det.bbox[2] - track.currentBox[2]) / dt,
                        (det.bbox[3] - track.currentBox[3]) / dt
                    ];
                }
                
                track.targetBox = det.bbox;
                track.confidence = det.confidence;
                track.lastSeen = now;
                track.missedFrames = 0;
            } else {
                track.missedFrames += 1;
            }
        });

        // 3. Create new tracks for unmatched detections
        newDets.forEach((det, idx) => {
            if (matchedDets.has(idx)) return;
            
            activeTracks.push({
                id: nextTrackId++,
                class: det.class,
                confidence: det.confidence,
                currentBox: [...det.bbox],
                targetBox: [...det.bbox],
                velocity: [0, 0, 0, 0],
                lastSeen: now,
                missedFrames: 0
            });
        });
    }

    function drawActiveTracks() {
        const ctx = resultCanvas.getContext('2d');
        ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

        const now = Date.now();
        const dt = now - lastAnimTime;
        lastAnimTime = now;

        if (dt <= 0) return;

        const classColors = {
            'car': '#10b981',
            'bus': '#ec4899',
            'truck': '#ef4444',
            'motorcycle': '#f59e0b',
            'motorbike': '#f59e0b',
            'bicycle': '#06b6d4',
            'person': '#3b82f6',
            'cow': '#8b5cf6',
            'dog': '#a78bfa'
        };

        // Filter out dead tracks (not seen for > 1500ms or missed too many frames)
        activeTracks = activeTracks.filter(track => {
            return (now - track.lastSeen < 1500) && (track.missedFrames < 15);
        });

        activeTracks.forEach(track => {
            const timeSinceLastDetection = now - track.lastSeen;
            
            // Update currentBox position
            for (let i = 0; i < 4; i++) {
                if (track.missedFrames === 0) {
                    // Smoothly interpolate towards target box
                    track.currentBox[i] += (track.targetBox[i] - track.currentBox[i]) * 0.15;
                } else {
                    // Extrapolate position using estimated velocity
                    track.currentBox[i] += track.velocity[i] * dt;
                }
            }

            const [x1, y1, x2, y2] = track.currentBox;
            const drawX = x1;
            const drawY = y1;
            const drawW = x2 - x1;
            const drawH = y2 - y1;

            if (drawW <= 0 || drawH <= 0) return;

            const color = classColors[track.class] || '#ffffff';
            
            ctx.save();
            // Fade out objects that haven't been detected recently
            if (timeSinceLastDetection > 500) {
                const alpha = Math.max(0, 1 - (timeSinceLastDetection - 500) / 1000);
                ctx.globalAlpha = alpha;
            }

            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.shadowBlur = 4;
            ctx.shadowColor = color;
            ctx.strokeRect(drawX, drawY, drawW, drawH);

            // Draw label background
            ctx.fillStyle = color;
            ctx.shadowBlur = 0;
            ctx.font = 'bold 12px sans-serif';
            const label = `${track.class} #${track.id} (${(track.confidence * 100).toFixed(0)}%)`;
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(drawX, drawY - 18, textWidth + 10, 18);

            // Draw text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, drawX + 5, drawY - 5);
            ctx.restore();
        });
    }
    function resetView() {
        stopLocalAnalysis();
        loadingSection.classList.add('hidden');
        resultSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        framesStat.classList.add('hidden');
        downloadVideoBtn.classList.add('hidden');
        resultImage.classList.remove('hidden');
        resultVideo.classList.add('hidden');
        resultCanvas.classList.add('hidden');
        resultImage.src = '';
        resultVideo.pause();
        resultVideo.src = '';
        fileInput.value = '';
    }

    updateModeUI();
    resetBtn.addEventListener('click', resetView);

    // 3D Card Hover Interaction
    const cards = document.querySelectorAll('.glass-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            // Calculate rotation degrees (-10 to 10 deg)
            const rotateX = -((y - centerY) / centerY) * 10;
            const rotateY = ((x - centerX) / centerX) * 10;
            
            card.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
            card.style.boxShadow = `0 20px 40px rgba(0, 0, 0, 0.4), var(--glass-shadow)`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'rotateX(0deg) rotateY(0deg) translateZ(0)';
            card.style.boxShadow = 'var(--glass-shadow)';
        });
    });
});
