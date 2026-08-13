"""
OpenAI-compatible serving entrypoint.
Run with vLLM once a checkpoint is merged and ready:

python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/codegen-v1-merged \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --port 8000
"""

if __name__ == "__main__":
    print("See module docstring for the vLLM serving command.")
