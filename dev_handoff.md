# Developer Handoff

## Introduction

This document provides a comprehensive overview of the F5-TTS project for new developers. It covers the project's architecture, codebase, and contribution guidelines.

## Getting Started

To set up your development environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SWivid/F5-TTS.git
    cd F5-TTS
    ```

2.  **Create a virtual environment:**
    ```bash
    conda create -n f5-tts python=3.11
    conda activate f5-tts
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

4.  **Install pre-commit hooks:**
    ```bash
    pip install pre-commit
    pre-commit install
    ```

## Codebase Overview

The project is organized into the following directories:

*   `src/f5_tts`: The main source code for the project.
    *   `model`: Contains the core model architecture, including the `CFM` module, `DiT` and `UNetT` backbones, and the `Trainer` class.
    *   `train`: Contains the scripts for training and finetuning the model.
    *   `infer`: Contains the scripts and utilities for running inference.
    *   `configs`: Contains the configuration files for the different models.
    *   `eval`: Contains the scripts for evaluating the model.
*   `tests`: Contains the unit tests for the project.
*   `data`: Contains the datasets used for training and evaluation.
*   `ckpts`: Contains the model checkpoints.

## Core Components

### `CFM`

The `CFM` class in `src/f5_tts/model/cfm.py` is the core of the F5-TTS model. It is a Conditional Flow Matching module that takes text and a reference audio snippet as input and generates a mel-spectrogram.

### `Trainer`

The `Trainer` class in `src/f5_tts/model/trainer.py` is responsible for training the `CFM` model. It uses the Hugging Face Accelerate library to handle multi-GPU training and provides a simple interface for training and finetuning the model.

### `F5TTS`

The `F5TTS` class in `src/f5_tts/api.py` provides a simple interface for using the F5-TTS model. It handles the loading of the model and vocoder, and provides methods for transcribing audio and generating speech.

## How to Contribute

1.  **Fork the repository:** Create a fork of the repository on GitHub.
2.  **Create a new branch:** Create a new branch for your changes.
3.  **Make your changes:** Make your changes to the codebase, following the coding style guidelines.
4.  **Run the pre-commit checks:** Before committing your changes, run the pre-commit checks to ensure that your code adheres to the project's coding standards:
    ```bash
    pre-commit run --all-files
    ```
5.  **Submit a pull request:** Submit a pull request to the `main` branch of the original repository.

## Future Work

*   **Improve the vocoder:** The current vocoder is based on Vocos, which is a good baseline but could be improved.
*   **Add support for more languages:** The current model is trained on English and Chinese, but it could be extended to support more languages.
*   **Explore different model architectures:** The current model is based on a Transformer backbone, but other architectures could be explored.
