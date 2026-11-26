FROM python:3.12-slim

WORKDIR /code

# install libraries first, so Docker can reuse this layer when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# then the code, the settings and the trained model
COPY params.yaml .
COPY src ./src
COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]