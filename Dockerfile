
FROM python:3.9-slim


WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


RUN python -m nltk.downloader punkt stopwords wordnet

EXPOSE 8000

ENV PYTHONUNBUFFERED=1


CMD ["uvicorn", "MAIN:app", "--host", "0.0.0.0", "--port", "8000"]
