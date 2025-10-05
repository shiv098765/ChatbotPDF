#PDF Chatbot Using Demo Photo

<img width="1211" height="859" alt="Screenshot 2025-10-05 115415" src="https://github.com/user-attachments/assets/61cace2c-7431-4b9f-9955-c898d8b70db9" />


<img width="1469" height="880" alt="Screenshot 2025-10-05 115303" src="https://github.com/user-attachments/assets/6793b4ae-27f8-4545-8af4-8c701fb038d0" />


# 📄 PDF Chatbot using Gemini Key

This project allows users to chat with their PDF files using Google’s Gemini model and LangChain.

##  How It Works
1. Upload a PDF file.
2. The app reads and splits the PDF into small text chunks.
3. It creates embeddings and stores them in a Chroma vector database.
4. Ask any question about the PDF — Gemini retrieves and answers based on the file’s content.

##  Tech Stack
- Python
- Streamlit
- LangChain
- ChromaDB
- Google Gemini API

# ⚙️ Run Locally
bash
pip install -r requirements.txt
streamlit run pdf_chatbot.py
