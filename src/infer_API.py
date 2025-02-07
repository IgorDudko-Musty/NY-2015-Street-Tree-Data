from fastapi import FastAPI, HTTPException
import uvicorn
from typing import List
import torch
from nnets import Model_Implementation


app = FastAPI()


@app.get('/')
def main_page():
    return "Pass to /doc and enter numbers to predict"


@app.post("/predict")
async def predict(data: List[float], paths: List[str], cl_amount: int):
    try:
        model = Model_Implementation(path_data_model=paths,
                                     mode='predict', 
                                     output_size=cl_amount, 
                                     device='cpu')
        data = torch.tensor(data).unsqueeze(0)
        predictions = model.predict(data) 
        return predictions
    except Exception as e: 
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__": 
    uvicorn.run(app, host="localhost", port=8002)