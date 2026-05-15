import base64
import hashlib
import io
import json
import threading
from pathlib import Path
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "source.ipynb"
DATASET_DIR = ROOT / "dataset"
IMAGE_DIR = DATASET_DIR / "images"
RESULT_DIR = ROOT / "results"
CHECKPOINTS = {
    "A1": ROOT / "checkpoints" / "best_vqa_a1_lstm.pt",
    "A2": ROOT / "checkpoints" / "best_vqa_a2_transformer.pt",
}
PRECOMPUTED_PREDICTIONS = {
    "B1": RESULT_DIR / "predictions_b1.json",
    "B2": RESULT_DIR / "predictions_b2_safe.json",
}

app = FastAPI(title="VQA Traffic Sign Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SquarePad:
    """Pad a PIL image to square without distorting the traffic sign shape."""

    def __init__(self, fill=(255, 255, 255)):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_side = max(width, height)
        pad_left = (max_side - width) // 2
        pad_top = (max_side - height) // 2
        pad_right = max_side - width - pad_left
        pad_bottom = max_side - height - pad_top

        from torchvision.transforms import functional as TF

        return TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)


class VQARequest(BaseModel):
    image_base64: str
    question: str
    model_name: str = "A2"


class ClassifyRequest(BaseModel):
    image_base64: str


class ExplainableVQARequest(VQARequest):
    pass


class DemoRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._definitions = None
        self._models: Dict[str, torch.nn.Module] = {}
        self._vocabs = {}
        self._image_hash_to_name = None
        self._prediction_cache = {}
        self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_definitions(self):
        if self._definitions is not None:
            return self._definitions

        if not NOTEBOOK_PATH.exists():
            raise RuntimeError(f"Missing notebook: {NOTEBOOK_PATH}")

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        core_source = None
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            if "class VQAModel" in source and "class SimpleVocab" in source:
                core_source = source
                break
        if core_source is None:
            raise RuntimeError("Could not find core VQA definitions in source.ipynb")

        namespace = {}
        exec(core_source, namespace)
        self._definitions = namespace
        return namespace

    def _prepare_transform(self) -> None:
        if self.transform is not None:
            return

        ns = self._load_definitions()
        transforms = ns["transforms"]
        self.transform = transforms.Compose(
            [
                SquarePad(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _fallback_vocabs_from_train_json(self):
        ns = self._load_definitions()
        SimpleVocab = ns["SimpleVocab"]
        train_path = DATASET_DIR / "train.json"
        if not train_path.exists():
            raise RuntimeError(f"Missing train split: {train_path}")

        train_records = json.loads(train_path.read_text(encoding="utf-8"))
        question_vocab = SimpleVocab.build([r["question"] for r in train_records], min_freq=2)
        answer_vocab = SimpleVocab.build([r["answer"] for r in train_records], min_freq=1)
        return question_vocab, answer_vocab

    def _vocabs_from_checkpoint(self, checkpoint: Dict):
        ns = self._load_definitions()
        SimpleVocab = ns["SimpleVocab"]

        question_token_to_id = checkpoint.get("question_token_to_id")
        answer_token_to_id = checkpoint.get("answer_token_to_id")
        if question_token_to_id and answer_token_to_id:
            return SimpleVocab(question_token_to_id), SimpleVocab(answer_token_to_id)

        # Old checkpoints did not save vocabularies; keep a fallback so the API
        # still runs, but retrained checkpoints should always include these maps.
        return self._fallback_vocabs_from_train_json()

    def get_model(self, model_name: str):
        model_key = "A1" if model_name.upper() == "A1" else "A2"
        with self._lock:
            if model_key in self._models:
                question_vocab, answer_vocab = self._vocabs[model_key]
                return self._models[model_key], model_key, question_vocab, answer_vocab

            self._prepare_transform()
            ns = self._load_definitions()
            build_vqa_model = ns["build_vqa_model"]
            decoder_type = "lstm" if model_key == "A1" else "transformer"
            checkpoint_path = CHECKPOINTS[model_key]

            if not checkpoint_path.exists():
                raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")
            if checkpoint_path.stat().st_size == 0:
                raise RuntimeError(f"Checkpoint is empty/corrupted: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            question_vocab, answer_vocab = self._vocabs_from_checkpoint(checkpoint)
            model = build_vqa_model(
                vocab_size=len(answer_vocab),
                decoder_type=decoder_type,
                pretrained_backbone=False,
                train_backbone=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device).eval()
            self._models[model_key] = model
            self._vocabs[model_key] = (question_vocab, answer_vocab)
            print(
                f"Loaded {model_key} from {checkpoint_path.name} "
                f"on {self.device} (epoch={checkpoint.get('epoch')}, "
                f"best_val_loss={checkpoint.get('best_val_loss')}, "
                f"q_vocab={len(question_vocab)}, a_vocab={len(answer_vocab)}, "
                f"checkpoint_vocab={'yes' if checkpoint.get('question_token_to_id') else 'no'})"
            )
            return model, model_key, question_vocab, answer_vocab

    def _build_image_hash_index(self) -> Dict[str, str]:
        if self._image_hash_to_name is not None:
            return self._image_hash_to_name

        image_hash_to_name = {}
        for path in IMAGE_DIR.iterdir():
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            image_hash_to_name[digest] = path.name
        self._image_hash_to_name = image_hash_to_name
        return image_hash_to_name

    def _load_precomputed_predictions(self, model_key: str) -> Dict:
        model_key = model_key.upper()
        if model_key in self._prediction_cache:
            return self._prediction_cache[model_key]

        path = PRECOMPUTED_PREDICTIONS[model_key]
        if not path.exists() and model_key == "B2":
            fallback = RESULT_DIR / "predictions_b2.json"
            if fallback.exists():
                path = fallback
        if not path.exists():
            raise RuntimeError(f"Missing precomputed predictions for {model_key}: {path}")

        rows = json.loads(path.read_text(encoding="utf-8"))
        by_image = {}
        for row in rows:
            by_image.setdefault(row["image"], []).append(row)
        payload = {"path": path, "by_image": by_image}
        self._prediction_cache[model_key] = payload
        return payload

    @staticmethod
    def _normalize_question(question: str) -> str:
        return " ".join(question.lower().strip().split())

    def predict_precomputed(self, image_bytes: bytes, question: str, model_name: str) -> Dict[str, str]:
        import difflib

        model_key = "B2" if model_name.upper() == "B2" else "B1"
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image_name = self._build_image_hash_index().get(image_hash)
        if image_name is None:
            raise RuntimeError(
                f"{model_key} demo uses saved predictions for dataset images only. "
                "Upload an image directly from dataset/images."
            )

        payload = self._load_precomputed_predictions(model_key)
        candidates = payload["by_image"].get(image_name, [])
        if not candidates:
            available_images = ", ".join(sorted(payload["by_image"].keys())[:12])
            raise RuntimeError(
                f"{model_key} currently has saved predictions only for the test split, "
                f"but image {image_name} is not in that prediction file. "
                f"Use one of these images for {model_key}: {available_images}, ..."
            )

        normalized_question = self._normalize_question(question)
        for row in candidates:
            if self._normalize_question(row["question"]) == normalized_question:
                return {
                    "answer": row["prediction"],
                    "model": model_key,
                    "image": image_name,
                    "reference": row.get("reference", ""),
                    "source": str(payload["path"]),
                }

        best = max(
            candidates,
            key=lambda row: difflib.SequenceMatcher(
                None,
                normalized_question,
                self._normalize_question(row["question"]),
            ).ratio(),
        )
        return {
            "answer": best["prediction"],
            "model": model_key,
            "image": image_name,
            "matched_question": best["question"],
            "reference": best.get("reference", ""),
            "source": str(payload["path"]),
        }

    @torch.no_grad()
    def predict_vqa(self, image: Image.Image, question: str, model_name: str) -> Dict[str, str]:
        model, model_key, question_vocab, answer_vocab = self.get_model(model_name)

        question_ids = question_vocab.encode(question)[:30]
        if not question_ids:
            question_ids = [question_vocab.bos_idx, question_vocab.eos_idx]

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        question_tensor = torch.tensor([question_ids], dtype=torch.long, device=self.device)
        question_lengths = torch.tensor([len(question_ids)], dtype=torch.long, device=self.device)

        generated = model.generate(image_tensor, question_tensor, question_lengths, max_length=12)
        answer = answer_vocab.decode(generated[0].detach().cpu().tolist())
        return {"answer": answer or "(khong sinh duoc cau tra loi)", "model": model_key}


runtime = DemoRuntime()


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def explainable_vqa_payload(answer: str, question: str, model_name: str) -> Dict[str, object]:
    answer_norm = normalize_text(answer)
    question_norm = normalize_text(question)

    if "cấm" in answer_norm or "không được" in answer_norm:
        category = "bien_cam"
        recommendation = "Không thực hiện hành vi bị cấm trên biển báo."
        explanation = "Câu trả lời chứa dấu hiệu cấm/hạn chế, nên hệ thống ưu tiên cảnh báo người tham gia giao thông."
    elif "nguy hiểm" in answer_norm or "cảnh báo" in answer_norm or "chú ý" in answer_norm:
        category = "bien_nguy_hiem"
        recommendation = "Giảm tốc độ và chú ý quan sát khu vực phía trước."
        explanation = "Câu trả lời thể hiện tình huống cảnh báo, vì vậy hành động an toàn là giảm tốc và quan sát."
    elif "hiệu lệnh" in answer_norm or "bắt buộc" in answer_norm or "phải" in answer_norm:
        category = "bien_hieu_lenh"
        recommendation = "Tuân thủ hướng hoặc hành động bắt buộc được thể hiện trên biển."
        explanation = "Câu trả lời có dấu hiệu hiệu lệnh/bắt buộc, nên người lái cần làm theo chỉ dẫn."
    elif "chỉ dẫn" in answer_norm or "hướng" in answer_norm or "đến" in answer_norm:
        category = "bien_chi_dan"
        recommendation = "Sử dụng thông tin trên biển để chọn đúng hướng di chuyển hoặc dịch vụ cần tìm."
        explanation = "Câu trả lời thiên về cung cấp thông tin chỉ dẫn, không phải cảnh báo hay cấm."
    else:
        category = "khong_ro"
        recommendation = "Quan sát thêm biển báo và ngữ cảnh xung quanh trước khi quyết định."
        explanation = "Câu trả lời chưa chứa đủ tín hiệu rõ ràng để suy ra nhóm biển báo chắc chắn."

    confidence = 0.78
    if not answer.strip() or "không rõ" in answer_norm:
        confidence = 0.45
    elif any(token in answer_norm for token in ["cấm", "chỉ dẫn", "nguy hiểm", "hiệu lệnh", "bắt buộc"]):
        confidence = 0.84
    if "ý nghĩa" in question_norm:
        confidence = max(confidence - 0.06, 0.35)

    return {
        "answer": answer,
        "model": model_name,
        "category": category,
        "explanation": explanation,
        "recommendation": recommendation,
        "confidence": round(confidence, 2),
    }


def decode_image_bytes(base64_str: str) -> bytes:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        return base64.b64decode(base64_str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}") from exc


def decode_image(base64_str: str) -> Image.Image:
    image_data = decode_image_bytes(base64_str)
    try:
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}") from exc


@app.get("/health")
async def health_endpoint():
    return {
        "status": "ok",
        "device": str(runtime.device),
        "checkpoints": {name: path.exists() for name, path in CHECKPOINTS.items()},
    }


@app.get("/", response_class=HTMLResponse)
async def demo_page():
    demo_path = ROOT / "demo.html"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing demo page: {demo_path}")
    return demo_path.read_text(encoding="utf-8")


@app.post("/vqa")
async def vqa_endpoint(request: VQARequest):
    image_bytes = decode_image_bytes(request.image_base64)
    image = decode_image(request.image_base64)
    try:
        if request.model_name.upper() in {"B1", "B2"}:
            return runtime.predict_precomputed(image_bytes, request.question, request.model_name)
        return runtime.predict_vqa(image, request.question, request.model_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/explainable-vqa")
async def explainable_vqa_endpoint(request: ExplainableVQARequest):
    image_bytes = decode_image_bytes(request.image_base64)
    image = decode_image(request.image_base64)
    try:
        if request.model_name.upper() in {"B1", "B2"}:
            prediction = runtime.predict_precomputed(image_bytes, request.question, request.model_name)
        else:
            prediction = runtime.predict_vqa(image, request.question, request.model_name)
        payload = explainable_vqa_payload(prediction["answer"], request.question, prediction["model"])
        payload.update(
            {
                "question": request.question,
                "reference": prediction.get("reference", ""),
                "matched_question": prediction.get("matched_question", ""),
                "source": prediction.get("source", ""),
            }
        )
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/classify")
async def classify_endpoint(request: ClassifyRequest):
    decode_image(request.image_base64)
    return {
        "category": "bien_bao",
        "vietnameseName": "Bien bao giao thong",
        "confidence": 0.95,
        "label": "vqa_demo",
        "description": "Endpoint phan loai dang dung mock; VQA endpoint da load checkpoint A1/A2 that.",
        "heatmapCenter": {"x": 0.5, "y": 0.5, "radius": 0.25},
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting VQA demo server at http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
