from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from PIL import Image as PILImage
import io
import torch
import numpy as np
import base64
import cv2
import os
from dotenv import load_dotenv
load_dotenv()
from transformers import AutoImageProcessor, SiglipForImageClassification
from openai import OpenAI

# ---- App setup ----

app = FastAPI(title="TrueSight AI vs Human Image Detector")

# Allow local Next.js dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client for GPT-based cue explanations
openai_client = OpenAI()

# ---- Model setup ----

MODEL_IDENTIFIER = "Ateeqq/ai-vs-human-image-detector"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TrueSight backend] Using device: {device}")

print(f"[TrueSight backend] Loading processor from: {MODEL_IDENTIFIER}")
processor = AutoImageProcessor.from_pretrained(MODEL_IDENTIFIER)

print(f"[TrueSight backend] Loading model from: {MODEL_IDENTIFIER}")
model = SiglipForImageClassification.from_pretrained(MODEL_IDENTIFIER)
model.to(device)
model.eval()
print("[TrueSight backend] Model and processor loaded successfully.")


# ---- Helper functions ----

def read_image(file_bytes: bytes) -> PILImage.Image:
    """Convert raw bytes into a RGB PIL image."""
    try:
        image = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e


def run_inference(image: PILImage.Image) -> Dict[str, Any]:
    """Run the model on a PIL image and return structured results."""
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Compute probabilities
    probabilities = torch.softmax(logits, dim=-1)[0]

    # Predicted class
    predicted_idx = int(torch.argmax(probabilities).item())
    id2label = model.config.id2label
    predicted_label = id2label[predicted_idx]

    scores: List[Dict[str, Any]] = []
    for i, label in id2label.items():
        score = float(probabilities[i].item())
        scores.append({"label": label, "score": score})

    return {
        "predicted_label": predicted_label,
        "predicted_index": predicted_idx,
        "confidence": float(probabilities[predicted_idx].item()),
        "scores": scores,
    }


def generate_gradcam_and_cues(
    image: PILImage.Image,
    target_layer: torch.nn.Module,
    class_idx: int,
    confidence: float,
):
    """
    Generate a Grad-CAM heatmap and derive simple cues from the heatmap and image.
    Returns (global_heatmap_base64, cues, cue_heatmaps).
    cue_heatmaps is a dict with separate overlays for Texture, Lighting, Background, Geometry.
    """
    # Preprocess and move to device
    inputs = processor(images=image, return_tensors="pt").to(device)
    image_tensor = inputs["pixel_values"]

    activations = []

    def forward_hook(module, inp, out):
        # out shape: (batch, seq_len, hidden_dim)
        activations.append(out)

    # Register hook
    forward_handle = target_layer.register_forward_hook(forward_hook)

    # Forward pass (no backward needed for this CAM approximation)
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits

    # Remove hook
    forward_handle.remove()

    if not activations:
        raise RuntimeError("No activations captured for Grad-CAM.")

    # Take first (and only) sample from batch
    activ = activations[0].detach().cpu()[0]  # (seq_len, hidden_dim)

    # Drop CLS token (first token), use patch tokens only
    if activ.shape[0] > 1:
        tokens = activ[1:, :]  # (num_patches, hidden_dim)
    else:
        tokens = activ

    # Compute L2 norm per patch as attention strength
    token_norms = torch.norm(tokens, dim=-1)  # (num_patches,)

    # Infer spatial size (assume square grid)
    num_patches = token_norms.shape[0]
    side = int(num_patches ** 0.5)
    if side * side != num_patches:
        # Fallback: just reshape to closest square grid
        side = int(np.sqrt(num_patches))
        side = max(1, side)
        cam = token_norms[: side * side].reshape(side, side)
    else:
        cam = token_norms.reshape(side, side)

    cam = cam.numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize CAM to image size
    img_np = np.array(image)
    h, w, _ = img_np.shape
    cam_resized = cv2.resize(cam, (w, h))

    # ---- Derive cues from Grad-CAM + image ----

    # Global attention stats
    global_mean = float(cam_resized.mean())
    # Define a central region (subject area)
    cy1, cy2 = int(h * 0.25), int(h * 0.75)
    cx1, cx2 = int(w * 0.25), int(w * 0.75)
    center_region = cam_resized[cy1:cy2, cx1:cx2]
    center_mean = float(center_region.mean()) if center_region.size > 0 else global_mean

    # Background = everything outside the center box
    background_mask = np.ones_like(cam_resized, dtype=bool)
    background_mask[cy1:cy2, cx1:cx2] = False
    if background_mask.any():
        background_mean = float(cam_resized[background_mask].mean())
    else:
        background_mean = global_mean

    # High-attention mask for local analysis
    high_mask = cam_resized > 0.6
    if high_mask.sum() < 10:
        high_mask = cam_resized > 0.4
    if high_mask.sum() < 10:
        high_mask = cam_resized > 0.2
    if high_mask.sum() == 0:
        high_mask = np.ones_like(cam_resized, dtype=bool)

    # Convert original image to grayscale for lighting/texture analysis
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) / 255.0

    # Lighting: variation (std) in brightness where the model looks
    global_brightness_std = float(gray.std() + 1e-6)
    att_brightness_std = float(gray[high_mask].std() + 1e-6)
    lighting_score = min(1.0, att_brightness_std / global_brightness_std) * confidence

    # Texture: edge strength (Canny) in high-attention regions
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    if edges.max() > 0:
        texture_strength = float(edges[high_mask].mean() / 255.0)
    else:
        texture_strength = 0.0
    texture_score = min(1.0, texture_strength * 2.0) * confidence

    # Geometry: how sharply the attention changes (gradient of CAM)
    grad_x = cv2.Sobel(cam_resized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cam_resized, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    if grad_mag.max() > 0:
        geometry_raw = float(grad_mag[high_mask].mean() / (grad_mag.max() + 1e-6))
    else:
        geometry_raw = 0.0
    geometry_score = min(1.0, geometry_raw) * confidence

    # Background cue: how much attention is leaking into the background
    if global_mean > 0:
        background_ratio = background_mean / global_mean
    else:
        background_ratio = 0.0
    # Higher ratio -> model looks relatively more at background
    background_score = float(min(1.0, background_ratio) * confidence)

    cues = [
        {"id": 1, "title": "Texture", "score": float(texture_score)},
        {"id": 2, "title": "Lighting", "score": float(lighting_score)},
        {"id": 3, "title": "Background", "score": float(background_score)},
        {"id": 4, "title": "Geometry", "score": float(geometry_score)},
    ]

    # ---- Build the global colored heatmap overlay ----
    def build_overlay_from_map(cam_like: np.ndarray) -> str:
        cam_norm = cam_like.copy()
        cam_norm = np.maximum(cam_norm, 0)
        if cam_norm.max() > 0:
            cam_norm = cam_norm / cam_norm.max()
        heatmap = (cam_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
        success, buffer = cv2.imencode(".png", overlay)
        if not success:
            raise RuntimeError("Failed to encode heatmap image.")
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    # Global Grad-CAM heatmap
    global_heatmap_base64 = build_overlay_from_map(cam_resized)

    # Per-cue maps derived from CAM + image features
    # Texture: emphasize edges where CAM is high
    edges_norm = edges.astype(np.float32) / 255.0
    texture_map = cam_resized * edges_norm

    # Lighting: emphasize brightness variation where CAM is high
    brightness_diff = np.abs(gray - gray.mean())
    if brightness_diff.max() > 0:
        brightness_norm = brightness_diff / brightness_diff.max()
    else:
        brightness_norm = brightness_diff
    lighting_map = cam_resized * brightness_norm

    # Background: focus on CAM in background regions
    background_map = cam_resized * background_mask.astype(np.float32)

    # Geometry: emphasize CAM where gradient magnitude is high
    if grad_mag.max() > 0:
        grad_norm = grad_mag / grad_mag.max()
    else:
        grad_norm = grad_mag
    geometry_map = cam_resized * grad_norm

    cue_heatmaps = {
        "texture": build_overlay_from_map(texture_map),
        "lighting": build_overlay_from_map(lighting_map),
        "background": build_overlay_from_map(background_map),
        "geometry": build_overlay_from_map(geometry_map),
    }

    return global_heatmap_base64, cues, cue_heatmaps


# ---- Cue explanation models ----

class CueExplanationRequest(BaseModel):
    cue_type: str  # e.g. "lighting", "geometry", "background", "texture"
    cue_title: str
    prediction_label: str  # "ai" or "hum"
    prediction_confidence: float  # 0–1
    cue_score: float  # 0–1
    image_context: Optional[str] = None
    # Optional base64-encoded images (e.g. data URLs) for multimodal GPT
    image_base64: Optional[str] = None        # original uploaded image
    heatmap_base64: Optional[str] = None      # optional cue-specific heatmap or global heatmap


class CueExplanationResponse(BaseModel):
    summary: str
    bullets: List[str]
    questions: List[str]


# ---- User cue explanation models ----

class UserCueExplanationRequest(BaseModel):
    image_base64: str
    user_observation: str
    model_prediction: Optional[str] = None
    model_confidence: Optional[float] = None


class UserCueExplanationResponse(BaseModel):
    summary: str
    bullets: List[str] = []
    questions: List[str] = []


# ---- StepConclusion explanation models ----

class ConclusionCuePayload(BaseModel):
    id: Any
    label: str
    score: Optional[float] = None
    source: Optional[str] = None  # "model", "user", or "both"
    note: Optional[str] = None


class ConclusionSummaryRequest(BaseModel):
    model_label: Optional[str] = None          # e.g. "ai", "human", "uncertain"
    model_probability: Optional[float] = None  # 0–1
    user_label: Optional[str] = None           # e.g. "ai", "human", "uncertain"
    user_confidence: Optional[float] = None    # 0–1, if collected
    agreement: Optional[bool] = None           # whether user & model agree
    cues: List[ConclusionCuePayload] = []


class ConclusionSummaryResponse(BaseModel):
    summary: str


# ---- API endpoints ----

@app.get("/")
async def root():
    return {
        "message": "TrueSight backend is running.",
        "model": MODEL_IDENTIFIER,
        "device": str(device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Basic prediction endpoint: returns prediction, confidence, and class scores.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload an image.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image = read_image(file_bytes)
    result = run_inference(image)

    return {
        "model": MODEL_IDENTIFIER,
        "prediction": result["predicted_label"],
        "confidence": result["confidence"],
        "scores": result["scores"],
    }


@app.post("/predict_with_explainability")
async def predict_with_explainability(file: UploadFile = File(...)):
    """
    Prediction endpoint with explainability:
    - base prediction (label, confidence, scores)
    - Grad-CAM heatmap (if generation succeeds)
    - heuristic cue scores based on model confidence
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload an image.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image = read_image(file_bytes)

    # Run base inference
    result = run_inference(image)
    predicted_label = result["predicted_label"]
    predicted_index = result["predicted_index"]

    heatmap_base64 = ""
    cues = []
    cue_heatmaps: Dict[str, str] = {}
    try:
        target_layer = model.vision_model.encoder.layers[-1]
        heatmap_base64, cues, cue_heatmaps = generate_gradcam_and_cues(
            image=image,
            target_layer=target_layer,
            class_idx=predicted_index,
            confidence=result["confidence"],
        )
    except Exception as e:
        print(f"[TrueSight backend] Grad-CAM generation failed: {e}")
        # No explainability available in this case.
        heatmap_base64 = ""
        cues = []
        cue_heatmaps = {}

    return {
        "model": MODEL_IDENTIFIER,
        "prediction": predicted_label,
        "confidence": result["confidence"],
        "scores": result["scores"],
        "heatmap": heatmap_base64,
        "cues": cues,
        "cue_heatmaps": cue_heatmaps,
    }



# ---- Cue explanation endpoint ----

@app.post("/explain_cue", response_model=CueExplanationResponse)
async def explain_cue(payload: CueExplanationRequest):
    """
    Generate a natural-language explanation for a single visual cue.
    This does NOT re-classify the image; it only explains how the cue
    might help a human interpret the model's decision.
    """

    cue_type = payload.cue_type.lower()
    prediction = payload.prediction_label.lower()
    confidence_pct = round(payload.prediction_confidence * 100, 1)
    score_pct = round(payload.cue_score * 100, 1)

    system_message = (
        "You are an assistant inside a visual analytics tool (TrueSight). Your job is to help a human decide whether an image "
        "is AI-generated or human-made by interpreting VISUAL CUES and (optionally) a model focus heatmap.\n"
        "\n"
        "CRITICAL RULES (must follow):\n"
        "1) NEVER describe what the image 'is' (no object labeling like 'this is a cheetah', 'this is a person'). Focus on VISUAL PROPERTIES only.\n"
        "2) NEVER claim the image IS AI-generated or IS real. Only discuss what observations might support or challenge the model’s prediction.\n"
        "3) Do NOT introduce any other classification task (e.g., human-vs-animal, human-like features, genre, scene type).\n"
        "4) Anchor your writing to the cue_type (lighting/texture/background/geometry) and the highlighted region. If no heatmap is present, refer to the most relevant region(s) for that cue.\n"
        "5) Keep it practical and inspection-oriented: tell the user WHAT TO LOOK AT, WHERE, and WHAT PATTERNS could be suspicious or benign.\n"
        "6) Stay neutral and avoid technical jargon.\n"
        "\n"
        "STYLE:\n"
        "- Short, concrete sentences.\n"
        "- No filler, no generic advice.\n"
        "- No bullet points unless explicitly asked (the output format below already specifies bullets for questions).\n"
    )

    user_message = (
        f"Context:\n"
        f"- Model prediction: '{prediction}' (confidence: {confidence_pct}%).\n"
        f"- Cue type: '{cue_type}' (title: '{payload.cue_title}').\n"
        f"- Model focus score for this cue: {score_pct}% (0–100).\n"
        f"- Input may include: the original image and optionally a heatmap overlay for this cue.\n\n"
        "TASK: Produce a 'How to read this cue' explanation for THIS specific image.\n\n"
        "OUTPUT FORMAT (must follow exactly):\n"
        "Write exactly 4 parts, in this order:\n"
        "A) WHERE TO LOOK (1 sentence): Point to the specific region(s) to inspect (use relative location words like 'upper-left', 'along the edge', 'in the blurred background behind the subject', etc.). If a heatmap exists, refer to the highlighted hotspot.\n"
        "B) WHAT TO CHECK (2 sentences): Describe concrete visual properties tied to the cue_type (lighting/texture/background/geometry). Mention 2–3 specific patterns to check for (e.g., repeated texture, overly smooth transitions, inconsistent blur, warped lines, unnatural edges), without naming objects.\n"
        "C) HOW IT COULD CUT BOTH WAYS (2 sentences): One sentence describing how these observations might SUPPORT the model prediction, and one sentence describing how they might CHALLENGE it. Stay neutral; do not assert truth.\n"
        "D) QUESTIONS (exactly 2 bullets): Provide exactly 2 short questions the user can ask themselves. Each question must be directly actionable and tied to the region and cue_type.\n\n"
        "ADDITIONAL CONSTRAINTS:\n"
        "- Do NOT mention animals, people, or object names.\n"
        "- Do NOT say 'human-like features' or anything similar.\n"
        "- Do NOT use the word 'definitely'.\n"
        "- Keep the total length under 120 words.\n"
    )
    if payload.image_context:
        user_message += (
            "\n\nAdditional context from the interface:\n"
            f"{payload.image_context}"
        )

    # Build multimodal user content: always include text, optionally include images
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": user_message}
    ]

    # If the frontend passes the original image as a base64 data URL, attach it
    if payload.image_base64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": payload.image_base64},
            }
        )

    # Optionally also attach a cue-specific or global heatmap overlay if available
    if payload.heatmap_base64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": payload.heatmap_base64},
            }
        )

    completion = openai_client.chat.completions.create(
        model="gpt-4o",  # multimodal-capable model
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    raw_text = completion.choices[0].message.content or ""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    summary_lines: List[str] = []
    bullet_lines: List[str] = []
    question_lines: List[str] = []

    for line in lines:
        lower = line.lower()
        if lower.startswith(("-", "*", "•")):
            bullet_lines.append(line.lstrip("-*• ").strip())
        elif lower.endswith("?"):
            question_lines.append(line)
        else:
            summary_lines.append(line)

    summary = " ".join(summary_lines).strip()
    if not summary:
        summary = raw_text.strip()

    return CueExplanationResponse(
        summary=summary,
        bullets=bullet_lines,
        questions=question_lines[:2],
    )


@app.post("/explain_user_cue", response_model=UserCueExplanationResponse)
async def explain_user_cue(payload: UserCueExplanationRequest):
    """
    Generate guidance for a user-provided observation using the uploaded image as context.
    """
    model_pred = payload.model_prediction or "unknown"
    model_conf = (
        f"{round(payload.model_confidence * 100, 1)}%"
        if payload.model_confidence is not None
        else "unknown"
    )

    system_message = (
        "You are an assistant inside a visual analytics tool (TrueSight). "
        "Your job is to help a user interpret their OWN suspicious observation in an image.\n"
        "- NEVER claim the image IS AI-generated or IS real; speak only in terms of visual cues and uncertainty.\n"
        "- Explain how to interpret the user's observation in plain language.\n"
        "- Provide concrete visual checks (what to look for) and what would support vs. weaken the suspicion.\n"
        "- Be concise. Return: 1 short paragraph summary, 2–4 bullet checks, and 2 reflective questions.\n"
    )

    user_message = (
        "Context:\n"
        f"- User observation: {payload.user_observation}\n"
        f"- Model prediction (optional context): {model_pred} (confidence: {model_conf})\n\n"
        "Task:\n"
        "Write a short explanation of how to read this observation as a cue. "
        "Include:\n"
        "1) Summary paragraph.\n"
        "2) Bullet checks (2–4) that a person can verify visually.\n"
        "3) Two short reflective questions.\n"
        "Do not mention training data or internal model details. Do not claim ground truth.\n"
    )

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": user_message}
    ]

    if payload.image_base64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": payload.image_base64},
            }
        )

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ],
    )

    raw_text = completion.choices[0].message.content or ""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    summary_lines: List[str] = []
    bullet_lines: List[str] = []
    question_lines: List[str] = []

    for line in lines:
        lower = line.lower()
        if lower.startswith(("-", "*", "•")):
            bullet_lines.append(line.lstrip("-*• ").strip())
        elif lower.endswith("?"):
            question_lines.append(line)
        else:
            summary_lines.append(line)

    summary = " ".join(summary_lines).strip()
    if not summary:
        summary = raw_text.strip()

    return UserCueExplanationResponse(
        summary=summary,
        bullets=bullet_lines,
        questions=question_lines[:2],
    )


# ---- StepConclusion explanation endpoint ----

@app.post("/explain_conclusion", response_model=ConclusionSummaryResponse)
async def explain_conclusion(payload: ConclusionSummaryRequest):
    """
    Generate a short narrative summary of how the model and user arrived at the final conclusion.

    This endpoint is used by the StepConclusion view to provide a human-readable explanation
    based on:
    - the model's label and confidence
    - the user's aggregated label and confidence
    - whether they agree or disagree
    - and the key cues that influenced the decision.
    """

    model_label = (payload.model_label or "").lower()
    user_label = (payload.user_label or "").lower()

    # Human-readable descriptions
    def format_label(label: str) -> str:
        if label == "ai":
            return "likely AI-generated"
        if label == "human":
            return "likely human-made"
        if label == "uncertain":
            return "uncertain"
        return label or "unknown"

    model_label_text = format_label(model_label)
    user_label_text = format_label(user_label) if user_label else "not provided"

    # Human-led final label: if the user provided a label, that is the operational conclusion.
    # Otherwise we fall back to the model's label.
    if user_label:
        final_label_text = user_label_text
    else:
        final_label_text = model_label_text

    model_conf_pct = (
        f"{round(payload.model_probability * 100, 1)}%" if payload.model_probability is not None else "unknown"
    )
    user_conf_text = (
        f"around {round(payload.user_confidence * 100, 1)}%" if payload.user_confidence is not None else "not recorded"
    )

    agreement_text: str
    if payload.agreement is True:
        agreement_text = "The user and the model agree on the final conclusion."
    elif payload.agreement is False:
        agreement_text = "The user and the model disagree on the final conclusion."
    else:
        agreement_text = "It is unclear whether the user and the model agree."

    # Build a concise textual representation of the key cues
    cue_lines: List[str] = []
    for cue in payload.cues:
        parts: List[str] = [f"- Cue '{cue.label}'"]
        if cue.score is not None:
            pct = round(cue.score * 100)
            parts.append(f"model focus ~{pct}%")
        if cue.source:
            parts.append(f"source: {cue.source}")
        if cue.note:
            parts.append(f"user note: {cue.note}")
        cue_lines.append(", ".join(parts))

    cues_block = "\n".join(cue_lines) if cue_lines else "(No specific cues were highlighted.)"

    system_message = (
        "You are an assistant inside a visual analytics tool (TrueSight) that explains how an AI image classifier "
        "and a human user each arrived at their own conclusions, and how these conclusions relate.\n"
        "- The AI model ALWAYS makes a binary prediction: either AI-generated or human-generated. The model is NEVER uncertain.\n"
        "- The human user evaluates visual cues and may conclude AI-generated, human-generated, or uncertain.\n"
        "- Clearly distinguish between the AI decision, the human decision, and the joint outcome.\n"
        "- Describe agreement, partial agreement, or disagreement explicitly as relationships between the AI and the human.\n"
        "- NEVER state or imply that the model was uncertain.\n"
        "- NEVER present the final outcome as purely human-led or purely model-led.\n"
        "- Use plain language, no bullet points, 3–6 sentences total.\n"
        "- Your answer MUST end with a separate final sentence that begins with exactly: 'Final joint conclusion:' "
        "followed by a description of the joint outcome (agreement, partial agreement, or disagreement).\n"
    )

    user_message = (
        "Here is the structured information about this case:\n\n"
        f"- AI decision: {model_label_text} (confidence: {model_conf_pct}).\n"
        f"- Human decision: {user_label_text} (self-reported confidence: {user_conf_text}).\n"
        f"- Joint outcome: {agreement_text}.\n\n"
        "Key cues considered by the human:\n"
        f"{cues_block}\n\n"
        "Please write a short narrative paragraph that:\n"
        "- explains what the AI model predicted and what the human concluded,\n"
        "- describes how the human used the cues to reach their conclusion,\n"
        "- explains how the AI and human conclusions relate (agreement, partial agreement, or disagreement),\n"
        "- briefly reflects on what this relationship implies for certainty or risk,\n"
        "- and ends with a final sentence starting exactly with 'Final joint conclusion:' followed by the joint outcome."
    )

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    raw_text = completion.choices[0].message.content or ""
    summary = raw_text.strip()
    if not summary:
        summary = (
            "A short summary could not be generated, but the structured information above still shows how "
            "the model and user combined their judgements based on the available cues."
        )

    return ConclusionSummaryResponse(summary=summary)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
