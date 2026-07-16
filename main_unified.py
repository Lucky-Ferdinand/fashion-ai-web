# File: main_unified.py (Master Terpadu - Klasifikasi Gambar, Next Item, Rating, & Dynamic Pricing - LAZY LOADING)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import io
import json
import pickle
import re
import traceback
import zipfile
import requests
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
from PIL import Image
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "all_models_data", "backend_dynamic"))

# =========================================================================
# FUNGSI PENGUNDUH OTOMATIS
# =========================================================================
def download_and_extract_models():
    target_check_path = os.path.join(BASE_DIR, "all_models_data", "backend_dynamic", "app", "inference.py")
    
    if not os.path.exists(target_check_path):
        print("⏳ File model tidak ditemukan secara lokal. Mengunduh dari Hugging Face (Mohon tunggu beberapa menit)...")
        output_zip = os.path.join(BASE_DIR, "all_models_data.zip")
        hf_url = "https://huggingface.co/datasets/Lucky1784/fashion-ai-models/resolve/main/all_models_data.zip?download=true"
        response = requests.get(hf_url, stream=True)
        
        if response.status_code == 200:
            with open(output_zip, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("📦 Unduhan selesai, mengekstrak file ke direktori proyek...")
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)
            if os.path.exists(output_zip):
                os.remove(output_zip)
            print("✅ Ekstraksi selesai!")
        else:
            raise Exception(f"Gagal mendownload dari Hugging Face. Status code: {response.status_code}")
    else:
        print("✅ Folder all_models_data lokal sudah tersedia.")

# =========================================================================
# VARIABEL GLOBAL (LAZY LOAD)
# =========================================================================
model = None
feature_extractor = None
df_csv = None
df_features = None
vision_loaded = False

next_item_model = None
item_encoder = None
metadata_dict = {}
next_item_loaded = False

rating_model = None
category_encoder = None
scaler_X = None
scaler_y = None
df_global = None
comparison_df = None
rating_loaded = False

forecast_service = None
dynamic_pricing_loaded = False

WOMEN_PATTERN = re.compile(r"\b(women|woman|wanita|ladies|girl|she)\b", re.IGNORECASE)
MEN_PATTERN = re.compile(r"\b(men|man|pria|boy|he)\b", re.IGNORECASE)
USD_TO_IDR = 17994

def guess_gender(title: str) -> str:
    if not title: return "unisex"
    if WOMEN_PATTERN.search(title): return "women"
    if MEN_PATTERN.search(title): return "men"
    return "unisex"

def format_idr(amount: float) -> str:
    rounded = int(round(amount / 1000.0)) * 1000
    return "Rp" + f"{rounded:,}".replace(",", ".")

def build_pricing(asin: str, raw_price) -> dict:
    try:
        base_usd = float(raw_price)
        if base_usd <= 0: raise ValueError
    except (TypeError, ValueError): base_usd = 12.0
    base_idr = base_usd * USD_TO_IDR
    on_sale = (abs(hash(asin)) % 5) == 0
    if on_sale:
        discount_percent = 30
        sale_idr = base_idr * (1 - discount_percent / 100)
        return {"on_sale": True, "discount_percent": discount_percent, "price": format_idr(sale_idr), "original_price": format_idr(base_idr)}
    return {"on_sale": False, "discount_percent": 0, "price": format_idr(base_idr), "original_price": None}

def forecast_future(model_inf, s_y, cat_encoded, current_seq, months=12):
    predictions = []
    future_seq = current_seq.copy()
    for _ in range(months):
        pred_scaled = model_inf.predict(future_seq.reshape(1, 3, 3), verbose=0)[0][0]
        pred_real = s_y.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(round(float(pred_real), 2))
        next_row = np.array([pred_scaled, future_seq[-1, 1], future_seq[-1, 2]])
        future_seq = np.vstack([future_seq[1:], next_row])
    return predictions

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128)) 
    import tensorflow as tf
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================================================================
# FUNGSI LAZY LOADERS
# =========================================================================
def load_vision_models():
    global model, feature_extractor, df_csv, df_features, vision_loaded
    if vision_loaded: return
    import tensorflow as tf
    print("⏳ [LAZY LOAD] Memuat model Vision & Rekomendasi ke RAM...")
    MODEL_PATH = os.path.join(BASE_DIR, 'all_models_data', 'backend_rekomendasi', 'model_klasifikasi_terbaik.keras')
    CSV_PATH = os.path.join(BASE_DIR, 'all_models_data', 'backend_rekomendasi', 'dataset_cv_final_v3.csv')
    PKL_PATH = os.path.join(BASE_DIR, 'all_models_data', 'backend_rekomendasi', 'database_fitur_rekomendasi_FULL.pkl')
    
    if os.path.exists(MODEL_PATH) and os.path.exists(CSV_PATH) and os.path.exists(PKL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        feature_extractor = tf.keras.Sequential(model.layers[:-1])
        df_csv = pd.read_csv(CSV_PATH)
        df_features = pd.read_pickle(PKL_PATH)
        df_features['Labels_lower'] = df_features['Labels'].astype(str).str.strip().str.lower()
        vision_loaded = True
        print("✅ Vision Model Loaded!")
    else:
        print("❌ File Vision Model tidak ditemukan.")

def load_next_item_models():
    global next_item_model, item_encoder, metadata_dict, next_item_loaded
    if next_item_loaded: return
    import tensorflow as tf
    print("⏳ [LAZY LOAD] Memuat model Next Item ke RAM...")
    try:
        # Menggunakan folder 'backend_next_item' (dengan huruf 't')
        MODEL_PATH = os.path.join(BASE_DIR, 'all_models_data', 'backend_next_item', 'fashion_gru_model.keras')
        ENCODER_PATH = os.path.join(BASE_DIR, 'all_models_data', 'backend_next_item', 'item_encoder.pkl')
        JSONL_PATH = os.path.join(BASE_DIR, 'all_models_data', 'backend_next_item', 'meta_Amazon_Fashion.jsonl')

        next_item_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(ENCODER_PATH, 'rb') as f:
            item_encoder = pickle.load(f)
        valid_asins = set(item_encoder.classes_)
        with open(JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    asin = data.get('parent_asin')
                    if asin and (asin in valid_asins):
                        images = data.get('images', [])
                        img_url = "https://placehold.co/480x600/f5f5f5/1a1a1a?text=UNIQLO"
                        if images and isinstance(images, list) and len(images) > 0: img_url = images[0].get('large', img_url)
                        title = data.get('title', 'Fashion Product')
                        metadata_dict[asin] = {"title": title, "image": img_url, "gender": guess_gender(title), **build_pricing(asin, data.get('price'))}
                except Exception: continue
        next_item_loaded = True
        print("✅ Next Item Model Loaded!")
    except Exception as e: print(f"⚠️ Load Warning Next Item: {e}")

def load_rating_models():
    global rating_model, category_encoder, scaler_X, scaler_y, df_global, comparison_df, rating_loaded
    if rating_loaded: return
    import tensorflow as tf
    print("⏳ [LAZY LOAD] Memuat model Rating LSTM ke RAM...")
    try:
        rating_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, activation="tanh", input_shape=(3,3), return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        rating_model.load_weights(os.path.join(BASE_DIR, "all_models_data/backend_rating/fashioncast_final.weights.h5"))
        category_encoder = joblib.load(os.path.join(BASE_DIR, "all_models_data/backend_rating/le_kategori.pkl"))
        scaler_X = joblib.load(os.path.join(BASE_DIR, "all_models_data/backend_rating/scaler_X.pkl"))
        scaler_y = joblib.load(os.path.join(BASE_DIR, "all_models_data/backend_rating/scaler_y.pkl"))
        df_global = pd.read_csv(os.path.join(BASE_DIR, "all_models_data/backend_rating/fashion_timeseries.csv"))
        comparison_df = pd.read_csv(os.path.join(BASE_DIR, "all_models_data/backend_rating/actual_vs_predicted.csv"))
        rating_loaded = True
        print("✅ Rating Model Loaded!")
    except Exception as e: print(f"⚠️ Load Warning Rating: {e}")

def load_dynamic_pricing_models():
    global forecast_service, dynamic_pricing_loaded
    if dynamic_pricing_loaded: return
    print("⏳ [LAZY LOAD] Memuat model Dynamic Pricing ke RAM...")
    try:
        app_path = os.path.join(BASE_DIR, "all_models_data", "backend_dynamic", "app")
        if app_path not in sys.path:
            sys.path.append(app_path)
            
        from inference import DemandForecastService
        artifact_path = os.path.join(BASE_DIR, "all_models_data", "backend_dynamic", "artifacts")
            
        forecast_service = DemandForecastService(artifact_dir=artifact_path)
        dynamic_pricing_loaded = True
        print("✅ Dynamic Pricing Model Loaded!")
    except Exception as e: print(f"⚠️ Load Warning Dynamic Pricing: {e}")

# =========================================================================
# LIFESPAN & APP INIT
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [STARTUP] API Berjalan. Memeriksa file model...")
    download_and_extract_models()
    print("✅ API Siap Menerima Request!")
    yield
    print("🛑 Mematikan server...")

app = FastAPI(title="Fashion AI Unified Master API - LAZY LOAD", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class HistoryRequest(BaseModel):
    click_history: list[str]
    gender: Optional[str] = "all"
class CategoryInput(BaseModel):
    category: str

# =========================================================================
# ENDPOINTS
# =========================================================================
@app.get("/")
def root():
    return {"status": "success", "message": "Fashion AI API Server is Running!"}

@app.post("/recommend")
def recommend_visual(file: UploadFile = File(...)):
    load_vision_models() 
    if not file: raise HTTPException(status_code=400, detail="File gambar tidak ditemukan dalam request")
    try:
        print("\n--- 🔍 MEMULAI PROSES KLASIFIKASI & REKOMENDASI ---")
        img_bytes = file.file.read()
        processed_img = preprocess_image(img_bytes)
        recommendations = []

        if model is None or feature_extractor is None or df_features is None or df_csv is None:
            return {"status": "error", "message": "Model belum termuat"}

        preds = model.predict(processed_img, verbose=0)
        class_idx = np.argmax(preds, axis=1)[0]
        labels = ["bag", "dress", "jacket", "pants", "shirt", "shoes", "skirt", "socks"] 
        predicted_category = labels[class_idx] if class_idx < len(labels) else "UMUM"
        
        query_embedding = feature_extractor.predict(processed_img, verbose=0).flatten()
        filtered_df = df_features[df_features['Labels_lower'] == predicted_category.lower()].copy()

        if len(filtered_df) > 0:
            db_embeddings_matrix = np.stack(filtered_df['embedding'].values)
            n_neighbors = min(5, len(filtered_df))
            knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
            knn.fit(db_embeddings_matrix)
            distances, indices = knn.kneighbors([query_embedding])
            
            for i, idx in enumerate(indices[0]):
                img_path = str(filtered_df.iloc[idx]['image_path'])
                parent_asin = img_path.replace('\\', '/').split('/')[-1].split('.')[0].strip()
                product_detail = df_csv[df_csv['parent_asin'].astype(str).str.strip() == parent_asin]
                
                if not product_detail.empty:
                    row = product_detail.iloc[0]
                    recommendations.append({
                        "parent_asin": str(row['parent_asin']).strip(), 
                        "title": str(row['title']).strip(),
                        "kategori": str(row.get('kategori', predicted_category)),
                        "gender": str(row.get('gender', 'UNISEX')),
                        "warna": str(row.get('warna', '-')),
                        "image_url": str(row.get('image_url', '')),
                        "similarity": float(1.0 - distances[0][i])
                    })
        return {"status": "success", "data": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products")
def get_catalog(page: int = 1, limit: int = 24, gender: Optional[str] = "all"):
    load_next_item_models()
    if not item_encoder: return {"products": [], "total_pages": 1, "total_items": 0}
    valid_catalog = []
    for asin in item_encoder.classes_:
        meta = metadata_dict.get(asin)
        if not meta: continue
        if gender and gender != "all" and meta["gender"] != gender: continue
        valid_catalog.append({"asin": asin, **meta})
    total_items = len(valid_catalog)
    total_pages = max((total_items + limit - 1) // limit, 1)
    start_idx = (page - 1) * limit
    return {"products": valid_catalog[start_idx:start_idx + limit], "current_page": page, "total_pages": total_pages, "total_items": total_items}

@app.post("/api/recommend")
def recommend_next_item(data: HistoryRequest):
    load_next_item_models()
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    if not data.click_history or not item_encoder: return []
    encoded_history = [int(item_encoder.transform([asin])[0]) + 1 for asin in data.click_history if asin in item_encoder.classes_]
    if not encoded_history: return []
    X_input = pad_sequences([encoded_history], maxlen=5, padding='pre')
    predictions = next_item_model.predict(X_input, verbose=0)[0]
    target_gender = (data.gender or "all").lower()
    if target_gender not in ("all", "women", "men", "unisex"): target_gender = "all"
    sorted_idx = np.argsort(predictions)[::-1]
    max_score = float(predictions.max()) if predictions.max() > 0 else 1.0
    recommendations = []
    for idx in sorted_idx:
        if idx == 0: continue
        try: asin_pred = item_encoder.inverse_transform([idx - 1])[0]
        except: continue
        meta = metadata_dict.get(asin_pred)
        if not meta: continue
        if target_gender != "all" and meta["gender"] != target_gender: continue
        recommendations.append({"asin": asin_pred, "match_score": round(float(predictions[idx]) / max_score * 100, 1), **meta})
        if len(recommendations) == 5: break
    return recommendations

@app.get("/categories")
def get_categories(): 
    load_rating_models()
    return {"categories": sorted(df_global["kategori"].unique().tolist()) if df_global is not None else []}

@app.get("/model-performance")
def model_performance(): 
    load_rating_models()
    return {"mae": 0.2869, "rmse": 0.4117, "mape": 7.28, "accuracy": 92.72, "comparison": comparison_df.to_dict(orient="records") if comparison_df is not None else []}

@app.post("/predict")
def predict_rating(data: CategoryInput):
    load_rating_models()
    if df_global is None or rating_model is None: return {"error": "Model atau dataset belum siap"}
    category = data.category
    if category not in df_global["kategori"].unique(): return {"error": "Kategori tidak ditemukan"}
    subset = df_global[df_global["kategori"] == category].copy().sort_values("date")
    subset["kategori_encoded"] = category_encoder.transform(subset["kategori"])
    category_encoded = int(subset["kategori_encoded"].iloc[-1])
    historical_data = [{"date": row["date"], "rating": round(float(row["rating"]), 2)} for _, row in subset.iterrows()]
    last_data = subset.tail(3)
    current_rating = float(last_data["rating"].iloc[-1])
    X_raw = last_data[["rating", "price", "kategori_encoded"]].values
    X_scaled = scaler_X.transform(X_raw)
    prediction = rating_model.predict(X_scaled.reshape(1, 3, 3), verbose=0)
    forecast_rating = scaler_y.inverse_transform(prediction)[0][0]
    trend = "increase" if forecast_rating > current_rating else ("decrease" if forecast_rating < current_rating else "stable")
    future_forecast = forecast_future(rating_model, scaler_y, category_encoded, X_scaled, months=12)
    return {
        "category": category, "current_rating": round(float(current_rating), 2), "forecast_rating": round(float(forecast_rating), 2),
        "trend": trend, "historical_data": historical_data, "future_forecast": future_forecast,
        "dynamic_pricing": {"action": "increase_price" if trend == "increase" else ("decrease_price" if trend == "decrease" else "keep_price"), "percentage": 5 if trend != "stable" else 0},
        "promotion_agent": {"action": "increase_promotion" if trend == "decrease" else "normal_promotion"},
        "inventory_agent": {"action": "increase_stock" if trend == "increase" else ("reduce_stock" if trend == "decrease" else "maintain_stock")}
    }

@app.post("/predict-demand")
def predict_demand_and_pricing(request: dict):
    load_dynamic_pricing_models()
    try:
        from pricing import make_dynamic_pricing_recommendation
        if not forecast_service: raise HTTPException(status_code=500, detail="DemandForecastService belum diinisialisasi.")
        prediction_result = forecast_service.predict(request)
        pricing_result = make_dynamic_pricing_recommendation(base_price=request.get("price", 12.0), historical_mean_demand_12m=prediction_result["historical_mean_demand_12m"], predicted_mean_demand_12m=prediction_result["predicted_mean_demand_12m"])
        history = [{"month": str(row["month"])[:7], "monthly_review_count": float(row["monthly_review_count"]), "monthly_avg_rating": float(row["monthly_avg_rating"])} for _, row in prediction_result["history_df"].iterrows()]
        return {
            "product": {"parent_asin": request.get("parent_asin"), "title": request.get("title"), "price": request.get("price"), "average_rating": request.get("average_rating"), "rating_number": request.get("rating_number")},
            "history": history, "forecast": prediction_result["forecast"], "pricing": pricing_result,
            "model_info": {"model_name": "GRU Delta Demand 12 Bulan", "target": "future_delta = future_log_demand", "forecast_horizon": 12}
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
