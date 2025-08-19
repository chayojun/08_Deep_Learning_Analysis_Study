from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as trnasforms
import torchvision.models as models
import json

app = FastAPI(title="ResNet34 inference")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet34(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=3, bias=True)
model.load_state_dict(torch.load('./model/mymodel.pth'))
model.eval()
model.to(device)


trnasforms_infer = trnasforms.Compose(
    [
        trnasforms.Resize((224,224)),
        trnasforms.ToTensor(),  
        trnasforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # 평균, 표준편차
    ]
)

class response(BaseModel):
    name : str
    score : float
    type : int

@app.post("/predict", response_model=response)
async def predict(file: UploadFile=File(...)):  # 무조건 키 값을 명시해야 한다. 상대방이 나에게 파일을 보낼때는 항상 키 값을 적어야한다.
      image = Image.open(file.file)
      image.save("./imgdata/test.jpg")  # 파일이름 정할 때: uuid, 카운트, timestamp
      img_tensor = trnasforms_infer(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

      with torch.no_grad():
        pred = model(img_tensor)
        print("예측값 : ", pred)
           
        pred_result = torch.max(pred,dim=1)[1].item() # 나오는 숫자: 0, 1, 2
        score = nn.Softmax()(pred)[0] # 나오는 %: [0.03, 0.09, 0.07]
        print("Softmax :", score)
        classname = ["마동석", "카리나", "이수지"]
        name = classname[pred_result]
        print("name :", name)
        
        return response(name=name, score=float(score[pred_result]), type=pred_result)
      

