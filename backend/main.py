from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
import pandas as pd
import io
import numpy as np
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from .preprocess import preprocess_data  # ✅ Import preprocessing function

app = FastAPI()

dataset = None  # Store dataset in memory


@app.get("/ping")
def ping():
    """Health check endpoint to verify if backend is running."""
    return {"message": "pong"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV/XLSX file, preprocess it, and store it in memory."""
    global dataset
    try:
        filename = file.filename.strip()  # Remove leading/trailing spaces

        if not filename.endswith((".csv", ".xlsx")):
            raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported.")

        contents = await file.read()

        # Read file based on format
        if filename.endswith(".csv"):
            dataset = pd.read_csv(io.BytesIO(contents), keep_default_na=False, na_values=["", "NA"], low_memory=False)
        else:  # XLSX
            dataset = pd.read_excel(io.BytesIO(contents), keep_default_na=False, na_values=["", "NA"])

        # ✅ Preprocess dataset before storing
        dataset = preprocess_data(dataset)

        return JSONResponse(content={
            "message": "File uploaded and preprocessed successfully!",
            "shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
            "preview": jsonable_encoder(dataset.head(5).to_dict(orient="records"))  # Show first 5 rows
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


def get_dataset():
    """Helper function to ensure dataset is uploaded"""
    global dataset
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a file first.")
    return dataset


@app.get("/show-columns/")
def show_columns(rows: int = 10, cols: int = 10, dataset: pd.DataFrame = Depends(get_dataset)):
    """Return a preview of dataset columns"""
    try:
        preview_df = dataset.iloc[:rows, :cols]
        return jsonable_encoder(preview_df.to_dict(orient="records"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving columns: {str(e)}")


@app.get("/shape/")
def get_shape(dataset: pd.DataFrame = Depends(get_dataset)):
    """Return shape of dataset (rows, columns)"""
    return {"rows": dataset.shape[0], "columns": dataset.shape[1]}


@app.get("/download/")
def download_cleaned(dataset: pd.DataFrame = Depends(get_dataset)):
    """Return cleaned dataset as a CSV file"""
    try:
        stream = io.StringIO()
        dataset.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            iter([stream.getvalue()]),  
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating CSV: {str(e)}")
