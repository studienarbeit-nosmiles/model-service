# FastAPI Project

## Overview

This project is a FastAPI-based web application designed to serve machine learning models via API endpoints. It provides an easy-to-use REST API interface for model inference, along with any additional functionalities like data preprocessing, model management, and performance monitoring.

## Features

- **FastAPI Framework**: High-performance asynchronous framework.
- **Machine Learning Model Integration**: Serve models for prediction and inference.
- **Data Validation**: Request body validation using Python type hints and Pydantic models.
- **Automatic Documentation**: OpenAPI and Swagger UI automatically generated.
- **Asynchronous Request Handling**: Built-in support for handling multiple requests efficiently.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/studienarbeit-nosmiles/model-service.git
   cd model-service
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Development Tools (Optional)**:

   For testing, linting, and code formatting, install these tools:

   ```bash
   tbd
   ```

## Running the Application

1. **Start the FastAPI Server**:

   ```bash
   uvicorn main:app --reload
   ```

   This will start the server at `http://127.0.0.1:8000`.

2. **API Documentation**:

   Visit the interactive API documentation at:

   - **Swagger UI**: `http://127.0.0.1:8000/docs`
   - **Redoc**: `http://127.0.0.1:8000/redoc`

## Project Structure

```
/your-fastapi-project
│
├── app/
│   ├── main.py           # Main FastAPI app
│   ├── models/           # Data models (Pydantic)
│   ├── api/              # API routes
│   ├── utils.py          # Utility functions (e.g., preprocessing)
│   └── __init__.py
│
├── tests/                # Unit tests
│   └── __init__.py
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .env                  # Environment variables
```

## Environment Variables

Set environment variables in a `.env` file:

```
tbd
```
