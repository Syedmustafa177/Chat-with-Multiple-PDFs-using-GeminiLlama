# Chat with Multiple PDFs using GeminiLlama

This repository hosts the code for a Streamlit-based application that enables users to interact with multiple PDF documents using two different AI models, Gemini and Groq. The application is designed to provide flexibility by utilizing different AI services to ensure continued functionality even when token limitations are reached with one model.

![image](https://github.com/Syedmustafa177/Chat-with-Multiple-PDFs-using-GeminiLlama/assets/113262233/fc3620fa-2100-40db-a4fc-d82a9613c38d)


## Overview

The application allows users to upload PDF files, extracts the text, and interacts with the content through user-generated questions. It leverages two different AI models (Gemini from Google Generative AI and Groq using Llama models) to answer questions based on the PDF content. This dual-model approach ensures higher availability and robustness, particularly in scenarios where one model might exhaust its usage limits.

## Features

- **PDF Text Extraction**: Utilizes `PyPDF2` for reading and extracting text from uploaded PDF documents.
- **Text Splitting**: Splits extracted text into manageable chunks using `RecursiveCharacterTextSplitter` for efficient processing.
- **Vector Storage**: Employs `FAISS` for creating and querying vector embeddings of the text chunks.
- **AI-Powered Responses**: Uses `ChatGoogleGenerativeAI` and `Groq` for generating answers to user questions based on the PDF context.
- **Streamlit Interface**: Offers a user-friendly web interface for uploading PDFs, submitting questions, and viewing AI responses.

## Getting Started


https://github.com/Syedmustafa177/Chat-with-Multiple-PDFs-using-GeminiLlama/assets/113262233/09affa09-2d75-40e1-af16-c6bb8131ef3e



### Prerequisites

- Python 3.8 or newer
- Pip for Python package installation

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Syedmustafa177/Chat-with-Multiple-PDFs-using-GeminiLlama.git
