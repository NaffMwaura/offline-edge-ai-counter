/**
 * EggCount AI - Application Logic
 * Handles Model Inference, Canvas Manipulation, and UI States
 */

const UI = {
    fileInput: document.getElementById('fileInput'),
    dropzone: document.getElementById('dropzone'),
    analysisView: document.getElementById('analysisView'),
    canvas: document.getElementById('canvas'),
    eggCountLabel: document.getElementById('eggCount'),
    latencyText: document.getElementById('latencyText'),
    loadingOverlay: document.getElementById('loadingOverlay'),
};

const ctx = UI.canvas.getContext('2d');
let session = null;

/**
 * Initialize the ONNX Inference Session
 */
async function initEngine() {
    try {
        // Path to your model in the GitHub repo
        const modelPath = './models/yolov8n.onnx';
        
        // Use WebGL for GPU acceleration (Sub-1s requirement)
        session = await ort.InferenceSession.create(modelPath, { 
            executionProviders: ['webgl', 'wasm'],
            graphOptimizationLevel: 'all'
        });
        console.log("AI Engine: Online (WebGL Accelerated)");
    } catch (e) {
        console.warn("WebGL not available, falling back to WASM/CPU. Performance may vary.");
        // Fallback or error messaging
    }
}

// Start engine on load
initEngine();

/**
 * Event Listeners
 */
UI.dropzone.onclick = () => UI.fileInput.click();

UI.fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) handleImageProcessing(file);
};

/**
 * Image Analysis Workflow
 */
async function handleImageProcessing(file) {
    UI.dropzone.classList.add('hidden');
    UI.analysisView.classList.remove('hidden');
    UI.loadingOverlay.classList.remove('hidden');

    const img = new Image();
    img.src = URL.createObjectURL(file);
    
    img.onload = async () => {
        // 1. Prepare Canvas
        UI.canvas.width = img.width;
        UI.canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        const startTime = performance.now();

        // 2. Inference Placeholder
        // Logic: Convert canvas to Float32 Tensor -> session.run() -> Post-process
        
        // --- SIMULATED INFERENCE FOR INITIAL TEST ---
        // Once your model is in the /models folder, replace this block with real logic
        setTimeout(() => {
            const endTime = performance.now();
            const latency = Math.round(endTime - startTime);
            
            UI.loadingOverlay.classList.add('hidden');
            
            // Random result for UI preview
            const detectedCount = Math.floor(Math.random() * 6) + 24; 
            animateResultCount(detectedCount);
            UI.latencyText.innerText = `${latency}ms`;
            
            drawDetections(detectedCount);
        }, 450); 
    };
}

/**
 * Visual Helpers
 */
function animateResultCount(target) {
    let current = 0;
    const interval = setInterval(() => {
        if (current >= target) {
            clearInterval(interval);
        } else {
            current++;
            UI.eggCountLabel.innerText = current;
        }
    }, 25);
}

function drawDetections(count) {
    ctx.strokeStyle = '#2563eb'; // blue-600
    ctx.lineWidth = Math.max(2, UI.canvas.width / 300);
    
    for(let i=0; i<count; i++) {
        // Mocking logic - real YOLO results provide [x,y,w,h]
        const w = 60; const h = 60;
        const x = Math.random() * (UI.canvas.width - w);
        const y = Math.random() * (UI.canvas.height - h);
        
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = '#2563eb';
        ctx.fillRect(x, y - 20, 35, 20);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText('Egg', x + 5, y - 5);
    }
}