/**
 * EggCount AI - Production Logic
 * Authentic Approach: Specific Class Filtering & Robust NMS
 */

const UI = {
    fileInput: document.getElementById('fileInput'),
    dropzone: document.getElementById('dropzone'),
    resultCard: document.getElementById('resultCard'), 
    canvas: document.getElementById('canvas'),
    eggCountLabel: document.getElementById('eggCount'),
    latencyText: document.getElementById('latency'), 
};

const ctx = UI.canvas ? UI.canvas.getContext('2d') : null;
let session = null;
const MODEL_SIZE = 640; 

/**
 * Initialize the ONNX Inference Session
 */
async function initEngine() {
    try {
        const modelPath = './models/yolov8n.onnx';
        session = await ort.InferenceSession.create(modelPath, { 
            executionProviders: ['webgl'],
            graphOptimizationLevel: 'all'
        });
        console.log("AI Engine: Online (WebGL Accelerated)");
    } catch (e) {
        console.warn("Falling back to CPU/WASM.");
        try {
            session = await ort.InferenceSession.create('./models/yolov8n.onnx', { executionProviders: ['wasm'] });
        } catch (err) {
            console.error("Model not found in /models/yolov8n.onnx");
        }
    }
}

initEngine();

if (UI.dropzone) {
    UI.dropzone.onclick = () => UI.fileInput.click();
}

if (UI.fileInput) {
    UI.fileInput.onchange = (e) => {
        const file = e.target.files[0];
        if (file) handleImageProcessing(file);
    };
}

if (UI.resultCard) {
    UI.resultCard.onclick = () => {
        UI.resultCard.classList.add('hidden');
        UI.dropzone.classList.remove('hidden');
        UI.fileInput.value = ""; 
    };
}

/**
 * Image Analysis Workflow
 */
async function handleImageProcessing(file) {
    if (UI.dropzone) UI.dropzone.classList.add('hidden');
    if (UI.resultCard) UI.resultCard.classList.remove('hidden');

    const img = new Image();
    img.src = URL.createObjectURL(file);
    
    img.onload = async () => {
        if (!UI.canvas || !ctx) return;

        UI.canvas.width = img.width;
        UI.canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        if (!session) {
            alert("AI Engine still initializing. Please wait a moment and try again.");
            return;
        }

        const startTime = performance.now();

        try {
            const tensor = await prepareInput(img);
            const feeds = { images: tensor };
            const output = await session.run(feeds);
            
            // Authentic Processing: Filter for egg-like COCO classes (e.g., sports ball, bowl, apple)
            const results = processOutput(output.output0.data, img.width, img.height);

            const endTime = performance.now();
            
            if (UI.latencyText) {
                UI.latencyText.innerText = `Inference: ${Math.round(endTime - startTime)}ms`;
            }
            
            animateResultCount(results.length);
            drawDetections(results);
        } catch (error) {
            console.error("Inference failed:", error);
        }
    };
}

async function prepareInput(img) {
    const canvas640 = document.createElement('canvas');
    canvas640.width = MODEL_SIZE;
    canvas640.height = MODEL_SIZE;
    const ctx640 = canvas640.getContext('2d');
    
    // Use clear background to avoid artifacts
    ctx640.fillStyle = "black";
    ctx640.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    
    // Fit image to 640x640 letterbox
    const ratio = Math.min(MODEL_SIZE / img.width, MODEL_SIZE / img.height);
    const newWidth = img.width * ratio;
    const newHeight = img.height * ratio;
    ctx640.drawImage(img, (MODEL_SIZE - newWidth) / 2, (MODEL_SIZE - newHeight) / 2, newWidth, newHeight);
    
    const imageData = ctx640.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const { data } = imageData;
    const float32Data = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
    
    // Normalize and convert to NCHW format
    for (let i = 0; i < data.length / 4; i++) {
        float32Data[i] = data[i * 4] / 255.0; // R
        float32Data[i + MODEL_SIZE * MODEL_SIZE] = data[i * 4 + 1] / 255.0; // G
        float32Data[i + 2 * MODEL_SIZE * MODEL_SIZE] = data[i * 4 + 2] / 255.0; // B
    }
    return new ort.Tensor('float32', float32Data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

/**
 * Authentic YOLO Output Processing
 */
function processOutput(data, imgWidth, imgHeight) {
    let boxes = [];
    const confThreshold = 0.25; // Lower initial threshold to catch all potential eggs
    const iouThreshold = 0.45; 

    // YOLOv8 output: [1, 84, 8400] -> [cx, cy, w, h, class0, class1, ..., class79]
    for (let i = 0; i < 8400; i++) {
        // We look specifically for "round objects" in the COCO dataset that eggs might mimic
        // Class 32 (sports ball), 45 (bowl), 47 (apple), 49 (orange)
        const targetClasses = [32, 45, 47, 49];
        let maxScore = 0;
        
        targetClasses.forEach(clsIdx => {
            const score = data[(clsIdx + 4) * 8400 + i];
            if (score > maxScore) maxScore = score;
        });

        if (maxScore > confThreshold) {
            const cx = data[0 * 8400 + i] / MODEL_SIZE * imgWidth;
            const cy = data[1 * 8400 + i] / MODEL_SIZE * imgHeight;
            const w = data[2 * 8400 + i] / MODEL_SIZE * imgWidth;
            const h = data[3 * 8400 + i] / MODEL_SIZE * imgHeight;
            
            boxes.push({ x: cx - w/2, y: cy - h/2, w, h, score: maxScore });
        }
    }

    // NMS Algorithm
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
        const chosen = boxes.shift();
        result.push(chosen);
        boxes = boxes.filter(box => calculateIoU(chosen, box) < iouThreshold);
    }

    return result;
}

function calculateIoU(boxA, boxB) {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
    const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const areaA = boxA.w * boxA.h;
    const areaB = boxB.w * boxB.h;
    return interArea / (areaA + areaB - interArea);
}

function animateResultCount(target) {
    if (!UI.eggCountLabel) return;
    let current = 0;
    const duration = 500; // ms
    const stepTime = Math.max(10, duration / (target || 1));
    
    UI.eggCountLabel.innerText = "0";
    if (target === 0) return;

    const interval = setInterval(() => {
        if (current >= target) {
            UI.eggCountLabel.innerText = target;
            clearInterval(interval);
        } else {
            current++;
            UI.eggCountLabel.innerText = current;
        }
    }, stepTime);
}

function drawDetections(boxes) {
    if (!ctx) return;
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    boxes.forEach(box => {
        ctx.strokeRect(box.x, box.y, box.w, box.h);
    });
}