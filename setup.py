from setuptools import setup, find_packages

setup(
    name="RuleReasoner",
    version="0.0.0",
    description="RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling",
    author="BIGAI",
    packages=find_packages(
        include=[
            "src",
        ]
    ),
    install_requires=[
        "google-cloud-aiplatform",
        "latex2sympy2",
        "pylatexenc",
        "sentence_transformers",
        "tabulate",
        "flash_attn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
