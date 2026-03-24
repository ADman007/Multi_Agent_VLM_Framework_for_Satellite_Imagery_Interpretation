# ISRO InterIIT - Satellite Imagery Interpretation System

This repository contains the solution for the **ISRO Inter IIT 14.0** challenge, featuring a Unified Inference Engine for Satellite Imagery Interpretation. The system utilizes a multi-model architecture to handle detailed image captioning, binary classification, numeric estimation, semantic reasoning, and precise visual grounding on satellite imagery.

This project is specifically optimized for remote sensing challenges such as small object sizes, dense urban layouts, shadow ambiguity, and limited spatial reasoning in general-purpose vision-language models.

---

##  Problem Motivation

Satellite imagery differs significantly from natural images:

- Top-down (nadir) perspective
- Tiny and densely packed objects
- Visually similar structures (rooftops, shadows, storage tanks)
- Weak spatial reasoning in generic VLMs

GeoNLI addresses these limitations through domain-specific fine-tuning and a hybrid neuro-symbolic grounding pipeline.

---

##  Features

### 1. Caption Generation
- Produces concise, domain-aligned scene descriptions
- Avoids verbose or hallucinated outputs
- Optimized using metric-aware prompting and supervised fine-tuning

---

### 2. Binary VQA (Yes/No)
- Strict "yes" or "no" outputs
- Deterministic decoding
- Domain-adapted training for higher exact-match accuracy

---

### 3. Numeric VQA

Supports two types of numeric queries:

**A. Object Counting**
- Aircraft, vehicles, tanks, buildings, etc.
- Handled by a fine-tuned 4-bit VLM

**B. Geometric Estimation**
- Area, length, width, perimeter
- Uses grounding + deterministic geometry computation

Hybrid routing ensures correct task handling.

---

### 4. Semantic VQA
- Land-use and structure classification
- Functional interpretation of regions
- Short, stable free-form answers
- Evaluated via semantic similarity metrics

---

### 5. Visual Grounding
- Returns oriented bounding boxes (OBBs)
- Parses structured constraints from text
- Uses segmentation for candidate mask generation
- Applies deterministic spatial filtering

This reduces spatial hallucinations common in black-box VLM-only systems.

---

##  Architecture Overview

### --> Orchestrator
Central pipeline controller that:
- Parses input JSON
- Loads image + metadata
- Routes queries to task-specific modules

---

### --> Vision-Language Backbone

**Primary backbone:** Qwen VL (developed by Alibaba)

Features:
- 4-bit quantization (QLoRA)
- Task-specific LoRA adapters
- Memory-efficient inference

---

### --> Segmentation & Grounding

Spatial localization uses **SAM3** (Segment Anything Model 3).

Grounding flow:
1. VLM extracts subject + constraints
2. SAM3 generates candidate masks
3. Deterministic geometric reasoning filters candidates
4. Masks converted to Oriented Bounding Boxes

**Confidence scoring:**
```
confidence = 0.3 × VLM_decision + 0.7 × spatial_score
```

---

## ☁️ Deployment Setup

Designed for GPU cloud deployment (RunPod):

- 2× NVIDIA A40 GPUs (48GB VRAM each)
- Docker containerization
- FastAPI server
- PyTorch + Hugging Face ecosystem
- vLLM for high-throughput reasoning

**GPU Allocation:**
- GPU 0 → API + LoRA models + SAM3
- GPU 1 → FP16 reasoning model

---

## 📂 Project Structure

```
.
├── app/
│   ├── api.py                 # FastAPI server implementation (Port 8000)
│   ├── main.py                # Pipeline Orchestrator & CLI entry
│   ├── vllm_server.py         # vLLM Server Manager (Port 8001)
│   ├── config.py              # Configuration & Model Paths
│   ├── grounding.py           # Grounding logic (SAM3 + Spatial Reasoning)
│   ├── spatial_reasoning.py   # Geometric analysis submodule
│   ├── sam3/                  # Segment Anything Model 3 integration
│   └── ... (task modules: binary.py, caption.py, numeric.py, semantic.py)
├── README.md                  # This file
└── ...
```

## 🛠️ Installation & Setup

### 1. Environment Setup

Required dependencies are listed in `app/requirements.txt`. It is recommended to use a virtual environment or Docker container.

```bash
pip install -r app/requirements.txt
```

### 2. Download Model Weights

The system uses several large models locally. Run the setup script to download them to your workspace (default: `/workspace/models`).

```bash
python app/download_weights.py
```

Models used:
- **Qwen2.5-VL-7B-Instruct (4-bit)**: Optimized for general understanding.
- **Qwen2.5-VL-7B-Instruct (FP16)**: For complex reasoning and grounding via vLLM.
- **SAM3**: For precise segmentation and object grounding.

## 💻 Usage

### 1. API Server (Recommended)

To start the FastAPI server:

```bash
# Using the startup script (also starts SSH)
bash app/start.sh

# Or directly with uvicorn
cd app
uvicorn api:app --host 0.0.0.0 --port 8000
```

**API Endpoint:** `POST /predict`

### 2. Command Line Interface

You can run the inference engine directly on a JSON file.

```bash
python app/main.py --input path/to/input.json --output path/to/output.json
```

## 📝 Input Format

The system accepts JSON input following the Inter-IIT format:

```json
{
    "input_image": {    
        "image_id": "sample1.png",
        "image_url": "https://bit.ly/4ouV45l",
        "metadata": {
            "width": 512,
            "height": 512,
            "spatial_resolution_m": 1.57
        }
    },
    "queries": {
        "caption_query": {
            "instruction": "Generate a detailed caption..."
        },
        "grounding_query": {
            "instruction": "Locate aircrafts..."
        },
        "attribute_query": {
            "binary": { "instruction": "Is there a..." },
            "numeric": { "instruction": "How many..." },
            "semantic": { "instruction": "What is..." }
        }
    }
}
```

## ⚙️ Configuration

System path configurations and model IDs can be customized in `app/config.py`. 
For Low-VRAM environments, you can toggle `LOW_VRAM_MODE = True` in `app/main.py`.

---

##  Evaluation Metrics

| Task | Metric |
|------|--------|
| Captioning | BERT-BLEU4 |
| Binary | Exact Match |
| Numeric | Exponential Relative Error |
| Grounding | Count Penalty × Mean IoU |

Evaluation conducted on the VRSBench validation split only.

---


##  Limitations

- Limited support for SAR / infrared imagery
- Segmentation errors in low-contrast regions
- No grounded-reasoning RL training (DPO/GRPO)
- Concurrency limits under GPU constraints

---

## 📜 License

This project uses open-source models including Qwen VL and SAM3 under their respective licenses.

---

**Team_24 – Inter IIT Tech Meet 14.0**  
GeoNLI | ISRO Challenge Submission 🚀
