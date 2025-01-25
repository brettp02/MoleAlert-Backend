from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import torch
from torchvision import transforms
from .model_starter import model_pipeline

app = FastAPI()

@app.get("/")
def read_root():
    return {"Health Check": "OK"}

@app.post("/predict")
async def predict(image: UploadFile):
    content = await image.read()
    pil_image = Image.open(io.BytesIO(content))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = transform(pil_image).unsqueeze(0)

    # Move the tensor to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_image = tensor_image.to(device)

    # Pass the tensor to model pipeline
    result = model_pipeline(tensor_image)

    return {"result": result}
