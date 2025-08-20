FROM python:3.11-slim

WORKDIR /app

COPY requirement.txt .

RUN pip install -r requirement.txt
RUN pip install uvicorn fastapi
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY infermain.py .

COPY model/ /app/model/
COPY imgdata/ /app/imgdata/

EXPOSE 7070
CMD ["uvicorn", "infermain:app", "--host", "0.0.0.0", "--port", "7070"]






