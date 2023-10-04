## Pytorch Inference App
Basic app to run inference using a pytorch model.

## Usage
local run
```
python3 app.py
```

docker run
```
docker build -t pytorch-inference-app .
docker run -p 8055:8055 pytorch-inference-app
```
