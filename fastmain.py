from fastapi import FastAPI

app = FastAPI()

@app.get('/')    # http://127.0.0.1/
def read_root():
    return {"result" : "카리나", "score" : "0.98"}

# @app.get("/image")
# def make_image():
#     return {"result" : "생성완료"}

@app.get("/chatbot")
def chatbot():
    return {"result" : "안녕하세요"}


@app.get('/image')
def make_image():
    return {"result" : "정필마"}