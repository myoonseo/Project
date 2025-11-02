# app.py — Flask + MongoDB + ChromaDB + DINO(이미지) + CLIP(텍스트) + Favorites

import os, io, time, hashlib, uuid
import numpy as np
from datetime import datetime, timedelta
from PIL import Image

from flask import (
    Flask, render_template, request, send_from_directory, send_file,
    abort, make_response, url_for, jsonify
)
from werkzeug.utils import secure_filename

# ===== Torch / Vision =====
import torch
import torchvision.transforms as T

# ===== Transformers (CLIP) =====
from transformers import CLIPModel, CLIPProcessor

# ===== MongoDB =====
import pymongo
from pymongo import ReturnDocument
from bson import ObjectId, binary as bson_binary

# ===== ChromaDB =====
import chromadb


# -------------------- 설정 --------------------
DATA_DIR   = "image_folder"     # 검색 대상 이미지 폴더
UPLOAD_DIR = "uploads"          # 업로드 쿼리 저장 폴더
TOP_K      = 6

os.makedirs(UPLOAD_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- MongoDB --------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo = pymongo.MongoClient(MONGO_URI)
db = mongo["image_db"]
col_images = db["images"]       # 리사이즈 이미지 바이너리 저장
col_favs   = db["favorites"]    # 즐겨찾기 {user_id, mongo_id, filename, created_at}

# 고유성 인덱스(중복 방지)
col_images.create_index([("orig_path", 1)], unique=True, background=True)
col_favs.create_index([("user_id", 1), ("mongo_id", 1)], unique=True, background=True)


# -------------------- 공통 유틸 --------------------
def ensure_user_cookie(resp):
    user_id = request.cookies.get("vuid")
    if not user_id:
        user_id = str(uuid.uuid4())
        resp.set_cookie(
            "vuid", user_id,
            max_age=int(timedelta(days=365).total_seconds()),
            httponly=True, samesite="Lax"
        )
    return resp


def save_resized_to_mongo(img_path: str, size=(320, 240)) -> str:
    """원본 경로(orig_path) 기준 upsert → 이미지 썸네일을 MongoDB에 저장하고 _id 반환"""
    img = Image.open(img_path).convert("RGB").resize(size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    doc = {
        "filename": os.path.basename(img_path),
        "orig_path": os.path.abspath(img_path),
        "data": bson_binary.Binary(buf.getvalue()),
        "w": size[0], "h": size[1],
    }
    up = col_images.find_one_and_update(
        {"orig_path": doc["orig_path"]},
        {"$set": doc},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    return str(up["_id"])


def mongo_image_bytes(oid: str) -> bytes:
    doc = col_images.find_one({"_id": ObjectId(oid)})
    if not doc:
        raise FileNotFoundError("mongo id not found")
    return bytes(doc["data"])


# -------------------- DINO (이미지 임베딩) --------------------
_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device).eval()
_dino_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def dino_feat(pil_img: Image.Image) -> np.ndarray:
    with torch.no_grad():
        x = _dino_tf(pil_img).unsqueeze(0).to(device)
        v = _dino(x)[0].detach().cpu().numpy().astype("float32")
        v /= (np.linalg.norm(v) + 1e-12)
        return v


# -------------------- CLIP (텍스트/이미지 임베딩) --------------------
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
_clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# Prompt Ensemble
PROMPTS = [
    "{}", 
    "a photo of {}",
    "a scenic view of {}",
    "a high quality photograph of {}",
    "an aesthetic travel photo of {}",
    "landscape featuring {}",
    "nature photography of {}",
    "urban photography showing {}",
    "a beautiful view of {}",
    "detailed shot of {}",
    "tourist view of {}",
    "outdoor scene with {}",
    "cinematic photo of {}",
    "a breathtaking scene of {}"
]


def clip_image_feat(pil_img: Image.Image) -> np.ndarray:
    with torch.no_grad():
        inputs = _clip_proc(images=pil_img, return_tensors="pt").to(device)
        v = _clip_model.get_image_features(**inputs)
        v = v / v.norm(p=2, dim=-1, keepdim=True)
        return v[0].detach().cpu().numpy().astype("float32")

def clip_text_feat_ensemble(text: str) -> np.ndarray:
    vecs = []
    with torch.no_grad():
        for p in PROMPTS:
            t = p.format(text)
            inp = _clip_proc(text=[t], return_tensors="pt", padding=True).to(device)
            v = _clip_model.get_text_features(**inp)        # [1, D]
            v = v / v.norm(p=2, dim=-1, keepdim=True)
            vecs.append(v[0].detach().cpu().numpy())
    v = np.mean(np.stack(vecs, axis=0), axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype("float32")


# -------------------- ChromaDB --------------------
chroma_client = chromadb.PersistentClient(path="chroma_db")

def _list_images(folder, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """하위 폴더까지 재귀 스캔 + 경로 대소문자/중복 정리"""
    paths, exts_lower = [], {e.lower() for e in exts}
    for root, _, files in os.walk(folder):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts_lower:
                paths.append(os.path.join(root, fn))
    norm = {os.path.normcase(os.path.abspath(p)): p for p in paths}
    return sorted(norm.values())

def _stable_id(path: str) -> str:
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()

def index_incremental(rebuild=False):
    print("[Index] scan:", DATA_DIR)
    paths = _list_images(DATA_DIR)
    t0 = time.time()
    total = len(paths)

    # Chroma 컬렉션 가져오기
    dino_coll = chroma_client.get_or_create_collection("images_dino", metadata={"hnsw:space":"cosine"})
    clip_coll = chroma_client.get_or_create_collection("images_clip", metadata={"hnsw:space":"cosine"})

    # 이미 Chroma에 존재하는 ID 목록 가져오기
    existing_ids = set(dino_coll.get(limit=999999, include=["metadatas"]).get("ids", []))

    # 현재 폴더에 존재하는 파일의 id 집합
    current_ids = set()

    ids_new, metas_new, emb_dino_new, emb_clip_new = [], [], [], []

    for i, p in enumerate(paths, 1):
        cid = _stable_id(p)
        current_ids.add(cid)

        # 기존에 존재하면 스킵
        if not rebuild and cid in existing_ids:
            continue

        try:
            mongo_id = save_resized_to_mongo(p, size=(320,240))
            pil = Image.open(p).convert("RGB")

            v_d = dino_feat(pil).tolist()
            v_c = clip_image_feat(pil).tolist()

            meta = {"filename": os.path.basename(p), "orig_path": os.path.abspath(p), "mongo_id": mongo_id}

            ids_new.append(cid); metas_new.append(meta)
            emb_dino_new.append(v_d); emb_clip_new.append(v_c)

            print(f"[Index] update: {p}")

        except Exception as e:
            print(f"[Index] failed {p}: {e}")

    # 삭제 감지 → Chroma에서 제거
    if not rebuild:
        deleted = existing_ids - current_ids
        if deleted:
            try:
                dino_coll.delete(ids=list(deleted))
                clip_coll.delete(ids=list(deleted))
                print(f"[Index] removed deleted files: {len(deleted)}")
            except:
                pass

    # rebuild 옵션이면 전체 날리고 다시
    if rebuild:
        chroma_client.delete_collection("images_dino")
        chroma_client.delete_collection("images_clip")
        dino_coll = chroma_client.get_or_create_collection("images_dino", metadata={"hnsw:space":"cosine"})
        clip_coll = chroma_client.get_or_create_collection("images_clip", metadata={"hnsw:space":"cosine"})

    # 새로 추가할 임베딩 저장
    if ids_new:
        dino_coll.add(ids=ids_new, embeddings=emb_dino_new, metadatas=metas_new)
        clip_coll.add(ids=ids_new, embeddings=emb_clip_new, metadatas=metas_new)
        print(f"[Index] added new: {len(ids_new)}")
    else:
        print("[Index] no new images")

    print(f"[Index] done | elapsed {time.time()-t0:.2f}s")


# -------------------- Flask ---------------------
app = Flask(__name__)

@app.route("/")
def index():
    count = col_images.estimated_document_count()
    resp = make_response(render_template("index.html", count=count, top_k=TOP_K))
    return ensure_user_cookie(resp)

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/mongo/<oid>")
def serve_mongo(oid):
    try:
        data = mongo_image_bytes(oid)
    except Exception:
        abort(404)
    return send_file(io.BytesIO(data), mimetype="image/jpeg")


# -------- 이미지 업로드 → DINO 검색 --------
@app.route("/search_image", methods=["POST"])
def search_image():
    f = request.files.get("file")
    if not f or f.filename == "":
        return "이미지를 업로드하세요.", 400

    safe = secure_filename(f.filename)
    base, ext = os.path.splitext(safe)
    unique = f"{base}_{int(time.time())}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    f.save(save_path)

    q_img = Image.open(save_path).convert("RGB")
    q = dino_feat(q_img).tolist()

    coll = chroma_client.get_or_create_collection("images_dino", metadata={"hnsw:space":"cosine"})
    res = coll.query(query_embeddings=[q], n_results=TOP_K)

    results, max_sim = [], None
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i, meta in enumerate(metas, 1):
        sim = 1.0 - float(dists[i-1]) if dists else None
        max_sim = sim if (max_sim is None or (sim is not None and sim > max_sim)) else max_sim
        results.append({"rank": i, "mongo_id": meta["mongo_id"], "filename": meta["filename"], "score": sim})

    resp = make_response(render_template(
        "results_image.html",
        query_url=url_for("serve_upload", filename=unique),
        results=results, max_sim=max_sim
    ))
    return ensure_user_cookie(resp)


# -------- 텍스트 → CLIP 검색 (Prompt Ensemble) --------
@app.route("/search_text", methods=["POST"])
def search_text():
    text = request.form.get("query", "").strip()
    if not text:
        return "텍스트를 입력하세요.", 400

    q = clip_text_feat_ensemble(text).tolist()
    coll = chroma_client.get_or_create_collection("images_clip", metadata={"hnsw:space":"cosine"})
    res = coll.query(query_embeddings=[q], n_results=TOP_K)

    results, max_sim = [], None
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i, meta in enumerate(metas, 1):
        sim = 1.0 - float(dists[i-1]) if dists else None
        max_sim = sim if (max_sim is None or (sim is not None and sim > max_sim)) else max_sim
        results.append({"rank": i, "mongo_id": meta["mongo_id"], "filename": meta["filename"], "score": sim})

    resp = make_response(render_template("results_text.html", query=text, results=results, max_sim=max_sim))
    return ensure_user_cookie(resp)


# -------- 즐겨찾기 --------
@app.route("/like", methods=["POST"])
def like():
    user_id = request.cookies.get("vuid") or str(uuid.uuid4())
    data = request.get_json(silent=True) or {}
    mongo_id = data.get("mongo_id"); filename = data.get("filename", "")
    if not mongo_id:
        return jsonify({"ok": False, "msg": "mongo_id required"}), 400

    col_favs.update_one(
        {"user_id": user_id, "mongo_id": mongo_id},
        {"$setOnInsert": {"filename": filename, "created_at": datetime.utcnow()}},
        upsert=True
    )
    resp = jsonify({"ok": True})
    if "vuid" not in request.cookies:
        resp.set_cookie(
            "vuid", user_id,
            max_age=int(timedelta(days=365).total_seconds()),
            httponly=True, samesite="Lax"
        )
    return resp

@app.route("/unlike", methods=["POST"])
def unlike():
    user_id = request.cookies.get("vuid")
    if not user_id:
        return jsonify({"ok": True})
    data = request.get_json(silent=True) or {}
    mongo_id = data.get("mongo_id")
    if not mongo_id:
        return jsonify({"ok": False, "msg": "mongo_id required"}), 400
    col_favs.delete_one({"user_id": user_id, "mongo_id": mongo_id})
    return jsonify({"ok": True})

@app.route("/favorites")
def favorites():
    user_id = request.cookies.get("vuid")
    favs = [] if not user_id else list(col_favs.find({"user_id": user_id}, {"_id": 0}))
    resp = make_response(render_template("favorites.html", favs=favs))
    return ensure_user_cookie(resp)


# -------- 수동 재인덱싱 --------
@app.route("/reindex", methods=["POST"])
def reindex():
    index_incremental(rebuild=True)
    return "reindexed", 200

# ---------------
if __name__ == "__main__":
    # 프로덕션처럼 한 프로세스만 실행
    index_incremental(rebuild=False)  # 부팅 시 1회만 인덱싱하고
    print("[Boot] starting Flask on :5007")
    app.run(host="0.0.0.0", port=5007, debug=False)  # 디버그 꺼짐