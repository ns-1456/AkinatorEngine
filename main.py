"""Entry point: run the Akinator API server. Run from project root: python main.py"""
import os
import uvicorn

if __name__ == "__main__":
    # Ensure cwd is project root so data/ and paths resolve
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
