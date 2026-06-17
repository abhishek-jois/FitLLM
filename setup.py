from setuptools import setup, find_packages

setup(
    name="fitllm",
    version="0.1.0",
    description="Layer-sharding LLM inference and training system",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.40",
        "datasets",
        "safetensors",
        "psutil",
        "bitsandbytes",
        "accelerate",
        "peft",
        "tqdm",
        "wandb",
    ],
    extras_require={
        "acceleration": [
            "flash-attn",
            "liger-kernel",
            "xformers",
        ],
    },
    entry_points={
        "console_scripts": [
            "fitllm=fitllm.__main__:main",
        ],
    },
)
