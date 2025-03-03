FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ src/

COPY models/ models/

CMD ["python", "src/app.py"]


