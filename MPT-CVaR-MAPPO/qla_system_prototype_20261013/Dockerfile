FROM python:3.13.5-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir numpy==2.2.6 Pillow==11.1.0 && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0
COPY systemlab /app/systemlab
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ENTRYPOINT ["python", "-m", "systemlab"]
