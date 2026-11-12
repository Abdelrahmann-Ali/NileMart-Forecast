FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/src:/app PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m nilemart.train

# Default: the API. For the planner UI, override the command:
#   docker run -p 8501:8501 nilemart streamlit run app/streamlit_app.py --server.address=0.0.0.0
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
