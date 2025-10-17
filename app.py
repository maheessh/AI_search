import os
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import timm
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
import zipfile
import shutil
import google.generativeai as genai
import json

# --- CONFIGURATION ---
MODEL_NAME = 'efficientnet_b0'
METADATA_DIR = f'metadata-files/{MODEL_NAME}'
INDEX_FILE = os.path.join(METADATA_DIR, 'image_features_vectors.idx')
METADATA_FILE = os.path.join(METADATA_DIR, 'image_data_features.pkl')
BASE_IMAGE_DIR = 'static/images'
UPLOAD_DIR = 'uploads'
BASE_METADATA_CSV = 'metadata.csv'
USER_COLLECTION_DIR = 'user_collection/images'
USER_METADATA_CSV = 'user_images.csv'

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=os.environ.get("AIzaSyChknza2O4rpT-j6mwDoinQAqFI3lV6CIE"))
    print("Gemini API Key configured successfully.")
except Exception as e:
    print(f"Could not configure Gemini API Key. Error: {e}")

os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(BASE_IMAGE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(USER_COLLECTION_DIR, exist_ok=True)

class Search_Setup:
    def __init__(self, model_name=MODEL_NAME, pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained
        print("Loading Model...")
        base_model = timm.create_model(self.model_name, pretrained=self.pretrained)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()
        print(f"Model {model_name} loaded successfully.")

    def _extract(self, img):
        img = img.resize((224, 224)).convert('RGB')
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
        feature = self.model(x).data.numpy().flatten()
        return feature / np.linalg.norm(feature)

    def get_feature_for_image(self, image_path: str):
        try:
            img = Image.open(image_path)
            return self._extract(img)
        except FileNotFoundError:
            print(f"\033[93mWarning: Could not find image file: {image_path}. Skipping.\033[0m")
            return None
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def run_index(self):
        base_df = pd.DataFrame()
        user_df = pd.DataFrame()

        if os.path.exists(BASE_METADATA_CSV):
            base_df = pd.read_csv(BASE_METADATA_CSV)
            print(f"Found metadata for {len(base_df)} items in base collection.")
        
        if os.path.exists(USER_METADATA_CSV):
            user_df = pd.read_csv(USER_METADATA_CSV)
            print(f"Found metadata for {len(user_df)} items in user collection.")
        
        combined_df = pd.concat([base_df, user_df], ignore_index=True)
        
        if combined_df.empty:
            print("\n\033[91mNo metadata files found (metadata.csv or user_images.csv). Nothing to index.\033[0m")
            return

        features = []
        for img_path in tqdm(combined_df['image_path'], desc="Extracting Features"):
            feature = self.get_feature_for_image(img_path)
            features.append(feature)

        combined_df['features'] = features
        combined_df = combined_df.dropna(subset=['features']).reset_index(drop=True)

        if combined_df.empty:
            print("\n\033[91mError: No valid images were found to process.\033[0m")
            print("\033[91mPlease ensure images are in the correct locations ('static/images/' or 'user_collection/images/') and try again.\033[0m")
            return

        combined_df.to_pickle(METADATA_FILE)
        print(f"Combined metadata for {len(combined_df)} images saved to {METADATA_FILE}")
        
        features_matrix = np.vstack(combined_df['features'].values).astype(np.float32)
        index = faiss.IndexFlatL2(features_matrix.shape[1])
        index.add(features_matrix)
        faiss.write_index(index, INDEX_FILE)
        print(f"Faiss index saved to {INDEX_FILE}")

# --- FLASK APP ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
search_instance = None

def parse_query_with_llm(query: str) -> dict:
    if not query: return {}
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    You are an expert e-commerce search assistant for a jewelry store. Your task is to extract structured filter attributes from a user's search query. The available filter attributes are: "metal", "gemstone", and "style". Analyze the user's query and determine the value for each attribute. If an attribute is not mentioned, do not include it in the output. Your ONLY output must be a valid JSON object. User Query: "{query}" JSON Output:
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(json_text)
    except Exception as e:
        print(f"Error parsing LLM response: {e}\nRaw response: {response.text}")
        return {}

def perform_search_and_filter(source_image_path: str, filters: dict) -> list:
    # NEW: Add a guardrail to prevent searching without an index
    if not os.path.exists(INDEX_FILE):
        raise ValueError("Search index not found. Please run 'python app.py index' first.")
        
    index = faiss.read_index(INDEX_FILE)
    image_data = pd.read_pickle(METADATA_FILE)
    query_vector = search_instance.get_feature_for_image(source_image_path)
    if query_vector is None: raise ValueError('Could not process source image')
    query_vector = np.array([query_vector], dtype=np.float32)
    D, I = index.search(query_vector, k=50)
    initial_results = [image_data.iloc[i].to_dict() | {'distance': float(dist)} for i, dist in zip(I[0], D[0])]
    final_results = initial_results
    if filters:
        for key, value in filters.items():
            if key in image_data.columns:
                final_results = [res for res in final_results if str(res.get(key, '')).lower() == str(value).lower()]
    final_results = final_results[:10]
    for res in final_results:
        res.pop('features', None)
        res['image_path'] = res['image_path'].replace('\\', '/')
    return final_results

@app.route('/api/add_image', methods=['POST'])
def api_add_image():
    if 'image' not in request.files: return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    try:
        filepath = os.path.join(USER_COLLECTION_DIR, file.filename)
        file.save(filepath)
        new_entry = {
            'id': [f"user_{file.filename}"], 'title': [os.path.splitext(file.filename)[0].replace('_', ' ').title()],
            'category': ['user upload'], 'metal': ['unknown'], 'gemstone': ['unknown'], 'style': ['unknown'],
            'image_path': [filepath.replace('\\', '/')]
        }
        new_df = pd.DataFrame(new_entry)
        if not os.path.exists(USER_METADATA_CSV):
            new_df.to_csv(USER_METADATA_CSV, index=False)
        else:
            new_df.to_csv(USER_METADATA_CSV, mode='a', header=False, index=False)
        return jsonify({'success': True, 'message': f"'{file.filename}' added to your collection. Please re-run the indexer."})
    except Exception as e:
        return jsonify({'error': 'An internal server error occurred'}), 500

@app.route('/api/multisearch', methods=['POST'])
def api_multisearch():
    source_image_path = request.form['image_path']
    text_query = request.form.get('query', '')
    try:
        filters = parse_query_with_llm(text_query)
        results = perform_search_and_filter(source_image_path, filters)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_search', methods=['POST'])
def api_upload_search():
    file = request.files['image']
    text_query = request.form.get('query', '')
    try:
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)
        filters = parse_query_with_llm(text_query)
        results = perform_search_and_filter(filepath, filters)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_indexing_logic():
    print("--- Starting Indexing Process ---")
    
    download_success = False
    if not os.path.exists(BASE_IMAGE_DIR) or not os.listdir(BASE_IMAGE_DIR):
        print(f"Base image directory '{BASE_IMAGE_DIR}' is empty. Attempting to download...")
        url = "https://storage.googleapis.com/aai-on-prose-prod-us-central1-public/temp/jewelry_dataset.zip"
        zip_path = "jewelry_dataset.zip"
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                os.makedirs(BASE_IMAGE_DIR, exist_ok=True)
                zip_ref.extractall(BASE_IMAGE_DIR)
            os.remove(zip_path)
            extracted_folder = os.path.join(BASE_IMAGE_DIR, 'jewelry_dataset')
            if os.path.exists(extracted_folder):
                for item in os.listdir(extracted_folder):
                    shutil.move(os.path.join(extracted_folder, item), BASE_IMAGE_DIR)
                os.rmdir(extracted_folder)
            print("\033[92mSample images downloaded successfully.\033[0m")
            download_success = True
        except Exception as e:
            print(f"\033[93mWarning: Automatic download of sample images failed: {e}.\033[0m")
            print("\033[93mThe system will now rely on your personal image collection.\033[0m")
    else:
        print("Base image directory already contains data. Skipping download.")
        download_success = True

    has_user_images = os.path.exists(USER_METADATA_CSV) and not pd.read_csv(USER_METADATA_CSV).empty

    if not download_success and not has_user_images:
        print("\n\033[91mCRITICAL: The base collection failed to download and your personal collection is empty.\033[0m")
        print("\033[92mACTION REQUIRED: Please start the server (`python app.py run`), upload images using the 'Add to Collection' feature, and then run this indexer again (`python app.py index`).\033[0m")
        return

    st = Search_Setup()
    st.run_index()
    print("\033[92m\n--- Indexing Complete ---\033[0m")
    print("You can now run the server with: \033[1mpython app.py run\033[0m")

def run_server_logic():
    global search_instance
    print("--- Starting Server ---")
    
    # NEW: Allow server to run in a limited mode even if index doesn't exist
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        print(f"\033[93m\nWarning: Index files not found. Starting in 'Initial Setup Mode'.\033[0m")
        print("In this mode, search functionality is disabled.")
        print("\033[92mACTION: Use the 'Add to Collection' feature on the web page to upload your images.\033[0m")
        print("\033[92mWhen you are done, stop this server (CTRL+C) and run `python app.py index` to build the search index.\033[0m")
    
    search_instance = Search_Setup()
    print("\n\033[92mServer is ready. Open your jewelry_search.html file in a browser.\033[0m")
    app.run(port=5000, debug=False)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['index', 'run']:
        print("Usage: python app.py [index|run]")
        sys.exit(1)
    command = sys.argv[1]
    if command == 'index':
        run_indexing_logic()
    elif command == 'run':
        run_server_logic()

