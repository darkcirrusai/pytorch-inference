# write a docker file that serves a pytorch model
FROM python:3.11.6-bullseye

WORKDIR /app

COPY docker_requirements.txt .

RUN pip install --no-cache-dir -r docker_requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8055

CMD ["python3", "app.py"]