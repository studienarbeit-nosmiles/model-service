from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI()

# Define a simple route
@app.get("/")
def read_root():
    return {"message": "Welcome to your FastAPI app!"}

if __name__ == "__main__":
    import uvicorn
    # Run the app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
