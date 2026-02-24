"""
FastAPI application for LogPPredML microservice
"""

from typing import Dict, Any, Optional, List
import os

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from logp_pred_ml_handler import LogPMLHandler

app = FastAPI(
    title="LogPMLHandler Microservice",
    description="Molecular property prediction service using LogPMLHandler models",
    version="0.1.0"
)

#Initialize handler
handler = LogPMLHandler()

# --- Request/Response Models ---

class HealthResponse(BaseModel):
    status: str
    message: str

class SMILESRequest(BaseModel):
    smiles: str
    property: Optional[List[str]] = None

class PropertyResult(BaseModel):
    property: str
    status: str
    results: Optional[float] = None
    error: Optional[str] = None

class MultiSMILESResponse(BaseModel):
    smiles: str
    status: str
    results: Dict[str, PropertyResult]
    error: Optional[str] = None

class BatchSMILESResponse(BaseModel):
    filename: str
    requested_properties: str 
    total_smiles: int
    results: List[MultiSMILESResponse]

# --- Routes ---

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="Healthy",
        message="LogPMLHandler is running"
    )

@app.post("/smi", response_model=MultiSMILESResponse)
async def predict_property(request: SMILESRequest) -> MultiSMILESResponse:
    if not request.smiles or request.smiles.strip() == "":
        raise HTTPException(status_code=400, detail="SMILES string cannot be empty")
    
    # Use the property list from request, or default to all
    prop_list = request.property if request.property and len(request.property) > 0 else handler.AVAILABLE_PROPERTIES

    # Call the updated handler method
    result = handler.process_multiple_properties(request.smiles.strip(), prop_list)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return MultiSMILESResponse(**result)

@app.post("/upload-smi", response_model=BatchSMILESResponse)
async def upload_smiles_file(file: UploadFile = File(...), property: Optional[List[str]] = None) -> BatchSMILESResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")
    
    if not property or len(property) == 0:
        property = handler.AVAILABLE_PROPERTIES

    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        smiles_list = [line.strip() for line in text.split("\n") if line.strip()]

        if not smiles_list:
            raise HTTPException(status_code=400, detail="File contains no SMILES strings")
        
        # 1. Process batch through handler using the updated method name
        results = handler.process_multiple_properties_batch(smiles_list, property)

        # 2. Convert results to response models (Fixed the syntax error here)
        response_results = [MultiSMILESResponse(**r) for r in results]

        return BatchSMILESResponse(
            filename=file.filename,
            requested_properties=",".join(property),
            total_smiles=len(smiles_list),
            results=response_results
        )
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
    
    except Exception as e:
        # Changed to 500 because it's usually a processing error at this stage
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
# This should be outside the upload function
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "1800"))
    uvicorn.run(app, host="0.0.0.0", port=port)
