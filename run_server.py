import uvicorn

if __name__ == "__main__":
    print("=================================================================")
    print(" 🚀 Deepfake Sentinel - Multi-Modal API & Dashboard Starting")
    print(" 🌐 Web Dashboard: http://127.0.0.1:8000")
    print(" 📖 Interactive Swagger API Docs: http://127.0.0.1:8000/docs")
    print("=================================================================")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
