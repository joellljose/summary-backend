import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import fitz  # pymupdf
import os
from dotenv import load_dotenv
import re

app = Flask(__name__)
CORS(app)

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=API_KEY)
MODEL_NAME = 'gemini-2.5-flash'

def extract_text_from_drive(url):
    """Downloads PDF from a URL and extracts text."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            return text
        else:
            raise Exception(f"Download failed with status: {response.status_code}")
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

@app.route('/generate-summary', methods=['POST'])
def generate_summary():
    try:
        data = request.get_json()
        pdf_url = data.get('url', '')
        
        if not pdf_url:
             return jsonify({"error": "No PDF URL provided"}), 400

        print(f"Generating summary for: {pdf_url}...")
        
        # 1. Extract Text
        source_text = extract_text_from_drive(pdf_url)
        
        if not source_text.strip():
             return jsonify({"error": "Extracted text is empty"}), 400

        # 2. Call Gemini
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        prompt = f"""
        Act as an academic expert. 
        Summarize the following text into 3-5 concise, high-value bullet points suitable for quick revision.
        Focus on key concepts, definitions, and formulas.
        
        Text:
        {source_text[:15000]}
        """
        
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Summary Generation Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on port 5001 to avoid conflict with main backend
    app.run(host='0.0.0.0', port=5001, debug=True)
