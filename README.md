AI Visual Search Demo - Setup Guide
Follow these steps to set up and run the full-stack AI visual search application on your local machine.

Prerequisites
Python 3.8+ installed.

pip (Python package installer).

Step 1: Set Up the Project Folder
Create a new folder on your computer for this project (e.g., ai-search-demo).

Save all four files (app.py, requirements.txt, jewelry_search.html, README.md) into this new folder.

Step 2: Set Up the Python Environment
It is highly recommended to use a virtual environment to keep dependencies isolated.

Open your terminal or command prompt.

Navigate to your project folder:

cd path/to/ai-search-demo

Create a virtual environment:

python -m venv venv

Activate the virtual environment:

On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

Your terminal prompt should now show (venv).

Step 3: Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

This may take a few minutes as it needs to download PyTorch, Faiss, and other libraries.

Step 4: Index the Sample Images
Before you can run the search, you need to process the sample images and create the AI index file. The script will automatically download a small dataset of jewelry images for you.

Run the following command in your terminal:

python app.py index

You will see progress bars as it extracts features from the images. When it's done, you will see a message "--- Indexing Complete ---". This step only needs to be done once.

This command will create three new folders in your project directory:

static/images/: Contains the downloaded sample jewelry images.

metadata-files/: Contains the AI model's index file.

uploads/: Where your uploaded images will be temporarily stored.

Step 5: Run the Backend Server
Now, start the Flask web server.

python app.py run

Your terminal will show that the server is running on http://127.0.0.1:5000. Leave this terminal window open.

Step 6: Run the Frontend
Navigate to your project folder in your file explorer.

Open the jewelry_search.html file directly in your web browser (e.g., Chrome, Firefox, Edge).

You can now use the application! Upload an image of a ring, necklace, or earrings, and the AI backend will return the most visually similar items from the sample dataset.