# ppu-dolphin-asr-onnx

A collection of ONNX models for Dolphin Automatic Speech Recognition by DataoceanAI and Tsinghua University.

---

## Overview

**Dolphin** is a state-of-the-art multilingual, multitask ASR model developed through a collaboration between [Dataocean AI](https://github.com/DataoceanAI/Dolphin) and Tsinghua University.

### Key Features

- **40 Eastern Languages**: Supports languages across East Asia, South Asia, Southeast Asia, and the Middle East
- **22 Chinese Dialects**: Comprehensive Chinese dialect support
- **210,000+ Hours**: Trained on extensive proprietary and open-source datasets

---

## Resources

- **Original Repository**: [DataoceanAI/Dolphin](https://github.com/DataoceanAI/Dolphin)
- **Research Paper**: [arXiv:2503.20212](https://arxiv.org/abs/2503.20212)
- **Colab Notebook**: [Try it out](https://colab.research.google.com/drive/1DhM89hGVvA9u4aYEChUzwxLvVtJV8jGW?usp=sharing)

---

## Available Models

This repository provides multiple model variants optimized for different use cases:

### Model Variants

| Variant         | Description                                               |
| --------------- | --------------------------------------------------------- |
| `dolphin-base`  | Standard base model with full precision                   |
| `dolphin-small` | Lightweight variant for resource-constrained environments |

All of the models exported while using `Greedy Search`, not `Beam Search`. You can export your own in collab notebook link above.

### Quantization Options

Each variant is available in multiple formats:

- **No Quantization** - Full precision (highest accuracy)
- **FP16** - Half precision (balanced performance)
- **INT8** - 8-bit quantization (fastest inference)

Not all of the models are in `onnx` format, some are in `.ort` format.

### Architecture Support

- AMD64 Desktop
- ARM processors

### Tokenization

All models share the same tokenizer using `tokens.txt`.

---

## Usage

This model collection is designed for integration with the upcoming **ppu-voiceland** package.

---

## Acknowledgments

Special thanks to [DakeQQ/Automatic-Speech-Recognition-ASR-ONNX](https://github.com/DakeQQ/Automatic-Speech-Recognition-ASR-ONNX) for providing the conversion scripts that made these ONNX models possible.

---

## License

Please refer to the original [Dolphin repository](https://github.com/DataoceanAI/Dolphin) for licensing information.
