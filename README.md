```# 💎 AI Visual Search Demo

This is a full-stack AI visual search application that lets users upload jewelry images and find visually similar items from a collection using deep learning and FAISS indexing.

---

## 🔧 Prerequisites

Make sure the following are installed on your system:

- Python 3.8 or higher  
- `pip` (Python package installer)  
- Git (optional, for version control)  

---

## 📁 Step 1: Set Up the Project Folder

1. Create a new folder for the project (e.g., `ai-search-demo`).  
2. Save the following files into it:  
   - `app.py` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Flask backend)  
   - `requirements.txt` &nbsp;(Python dependencies)  
   - `jewelry_search.html` (Frontend UI)  
   - `README.md` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(This file)  

---

## 🐍 Step 2: Set Up the Python Environment

> It's strongly recommended to use a virtual environment to isolate project dependencies.

### 🔹 Open Terminal and Navigate to Your Project Directory:

```bash
cd path/to/ai-search-demo
🔹 Create a Virtual Environment:
bash
Copy code
python -m venv venv
🔹 Activate the Virtual Environment:
On Windows:

bash
Copy code
.\venv\Scripts\activate
On macOS/Linux:

bash
Copy code
source venv/bin/activate
Once activated, your terminal should show (venv) at the beginning of the line.

📦 Step 3: Install Dependencies
Install all required Python libraries from requirements.txt:

bash
Copy code
pip install -r requirements.txt
This will install PyTorch, FAISS, Flask, and other required packages. It may take a few minutes.

🖼️ Step 4: Index the Sample Images
This step builds the image index from the sample jewelry dataset.

🔹 Run the following command:
bash
Copy code
python app.py index
You’ll see progress bars and then a message:

diff
Copy code
--- Indexing Complete ---
This will automatically:

Download a small dataset of jewelry images.

Extract visual features from each image.

Create an AI-powered FAISS index.

📂 Folders Created:
static/images/ — Sample images

metadata-files/ — AI index + metadata

uploads/ — Temporary storage for uploads

🚀 Step 5: Run the Backend Server
Start the Flask web server:

bash
Copy code
python app.py run
Your terminal should show something like:

csharp
Copy code
 * Running on http://127.0.0.1:5000/
Leave this terminal running while using the app.

🌐 Step 6: Run the Frontend
Open your project folder in your file explorer.

Double-click on jewelry_search.html to open it in your web browser.

✅ You can now upload an image of a ring, necklace, or earrings, and see visually similar matches from the dataset.

📝 Optional: Add Your Own Images
You can add your own jewelry images to the dataset:

Use the “Add to Collection” feature in the web interface.

Then re-run the indexing step:

bash
Copy code
python app.py index
This updates the FAISS index with your new images.

❓ Troubleshooting
ModuleNotFoundError: Ensure you're using the virtual environment and dependencies are installed.

git : command not found: Make sure Git is installed and added to your system PATH.

No matches returned: Confirm that python app.py index completed successfully and that images exist in static/images/.

📄 License
This project is intended for educational and demo purposes only.

Ensure you have the rights to use any additional images or data you add.

🙋 Need Help?
Open an issue on the GitHub repository or reach out to the maintainer for support.

```
