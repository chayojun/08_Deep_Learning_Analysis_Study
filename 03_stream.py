import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from streamlit_drawable_canvas import st_canvas

class_names = ["마동석", "카리나", "이수지"]

# 모델 불러오기

@st.cache_resource
# 밑에 있는 함수를 캐쉬에 저장해서 사용하겠다는 뜻 불러오는 속도가 빨라서 효율적
def load_model():
    model = nn.Sequential(
        nn.Linear(784,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,10)
    )
    model.load_state_dict(torch.load("src/num_model.pth",map_location=torch.device('cpu')))
    # .load_state_dict: 가중치 불러오기
    model.eval()
    return model

#이미지 전처리
def transform_image(image):
    transforms_test = transforms.Compose(
        [
            transforms.Resize((28,28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ]
    )
    return transforms_test(image).unsqueeze(0) # (3,224,224)
    # .unsqueeze(0): 배치 사이즈를 추가 해준다.

st.title("숫자분류기 V.1")

# 파일을 업로드하는 작업



# camimg = st.camera_input("웹캠")


canvas_img = st_canvas(
    fill_color="white", # 내부 색상  #001100 , RGB, RGBA
    stroke_width=3,  # 펜두께
    stroke_color="black",
    background_color="white",
    height=400,
    width=400,
    drawing_mode="freedraw", # 모드 (freedraw, line, rect, circle, transform, poly)
    key = "canvas")
image = Image.fromarray(canvas_img.image_data).convert("RGB") 



if image is not None:
    
    # .convert("RGB"): 이미지를 열고, 혹시 다른 색상 모드여도 무조건 RGB로 맞춘다
    st.image(image,caption="업로드 이미지",use_container_width=True)
    # use_container_width=True: 원본 이미지 대로 불러오기
    # use_container_width=False: 크기를 streamlit 사이즈로 맞춰서 가지고 오기
    
    model = load_model() # 캐쉬에서 불러온 모델
    infer_img = transform_image(image) # 전처리 된 이미지

    with torch.no_grad():
        # infer_img = infer_img.view(-1,28,28) 
        result = model(infer_img)
        preds = torch.max(result,dim=1)[1]
        pred_classname = class_names[preds.item()]
        confidence = torch.softmax(result,dim=1)[0][preds.item()].item() * 100
        

    st.success(f"예측결과: **{pred_classname}** ({confidence:.2f})% 확신")