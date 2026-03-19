import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import ollama

app = FastAPI(
    title="Ollama Signature Verification API", 
    description="Independent Test API back-end to compare two signature images using Ollama vision models."
)

DEFAULT_PROMPT = """You are an expert forensic document examiner.
Please carefully analyze and compare the two provided signature images. Image 1 is the already existing (reference/genuine) signature, and Image 2 is the questioned signature to be verified.

Focus your analysis rigorously on the following key aspects:
1. Stroke Dynamics: Differences in stroke entry and exit angles.
2. Morphology: Loop size, spacing, and shape consistency.
3. Execution: Pen lift points (where strokes start/end), line quality, and pressure variations.
4. Structural Alignment: Overall slant, baseline alignment, and relative proportions.

Based on a meticulous comparison of these characteristics, please provide:
- A detailed step-by-step forensic explanation of your findings.
- A final calculated matching score ranging from 0 to 100 (where 100 indicates absolute certainty of an exact match in characteristics).
- A final conclusive verdict: "Genuine", "Suspicious", or "Likely Forged". 
- Provide a score between 0 and 100 in the response."""


@app.get("/api/models")
def get_available_models():
    """Returns a list of all available models from the local Ollama instance."""
    try:
        models_response = ollama.list()
        
        model_names = []
        # Support dict format (older python client) or object format (newer python client)
        if isinstance(models_response, dict) and 'models' in models_response:
            model_names = [m.get('name') for m in models_response['models']]
        elif hasattr(models_response, 'models'):
            model_names = [getattr(m, 'model', getattr(m, 'name', str(m))) for m in models_response.models]
            
        return {"status": "success", "models": model_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models from Ollama: {str(e)}")


@app.post("/api/compare")
async def compare_signatures(
    image_1: UploadFile = File(..., description="The reference (already existing/genuine) signature image"),
    image_2: UploadFile = File(..., description="The queried signature image to verify"),
    model: str = Form("qwen2.5vl:3b", description="The Ollama vision model to use (default: qwen2.5vl:3b)"),
    prompt: Optional[str] = Form(None, description="Optional custom master prompt. If not provided, a default professional forensic prompt is used.")
):
    """
    Compare two signature images using the chosen Ollama vision model.
    Pass `image_1` and `image_2` as multipart/form-data.
    You can dynamically select the `model` and optionally override the master `prompt`.
    """
    try:
        # Read the image files into bytes
        img1_bytes = await image_1.read()
        img2_bytes = await image_2.read()
        
        final_prompt = prompt if prompt and prompt.strip() else DEFAULT_PROMPT
        print(final_prompt[100])
        
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': final_prompt,
                'images': [img1_bytes, img2_bytes]
            }]
        )
        
        return {
            "status": "success",
            "model_used": model,
            "prompt_used": final_prompt,
            "result": response.get('message', {}).get('content', '')
        }
    except ollama.ResponseError as e:
        raise HTTPException(status_code=400, detail=f"Ollama API Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    # Start the test API server on port 8001
    print("Starting Signature Verification Test API on http://localhost:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)
