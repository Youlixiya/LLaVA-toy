python -m llava.serve.controller --host 127.0.0.1 --port 10000
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.model_worker --host 127.0.0.1 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/llava-siglip-tinyllama-v1.0-1.1b-3t-rec --load-4bit

CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli \
    --model-path ./checkpoints/llava-siglip-tinyllama-v1.0-1.1b-3t \
    --image-file "serve_images/2024-02-02/b939abf2c4553ce07e642170aee3a3d7.jpg" \
    --load-4bit