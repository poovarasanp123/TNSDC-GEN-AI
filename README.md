# TNSDC-Generative-AI-
# Handwritten Model using GAN (Generative Adversarial Network)

This repository contains code for training and generating handwritten digits using Generative Adversarial Networks (GANs).

## Overview

Generative Adversarial Networks (GANs) are a class of deep learning models used for generating new data samples from an existing dataset. In the context of handwritten digits, GANs can be trained on a dataset of handwritten digits (such as MNIST) and generate new, realistic-looking handwritten digits.

This project specifically focuses on training a GAN to generate handwritten digits resembling those found in the MNIST dataset.

## Requirements

- Python 3.x
- TensorFlow (>=2.0)
- NumPy
- Matplotlib

## Usage

1. **Dataset**: The code is designed to work with the MNIST dataset by default, which can be easily loaded using TensorFlow's built-in datasets module.

2. **Training**: Run the `handwritten model using GAN.ipynb` script to train the GAN model. You can specify hyperparameters such as batch size, number of epochs, learning rate, etc., through command-line arguments or by directly modifying the script.

    ```bash
    python handwritten model using GAN.ipynb --epochs 100 --batch_size 128 --learning_rate 0.0002
    ```

3. **Generating Handwritten Digits**: After training the model, you can generate handwritten digits using the `handwritten model using GAN.ipynb` script. Specify the number of digits you want to generate using the `--num_samples` argument.

    ```bash
    python handwritten model using GAN.ipynb --num_samples 10
    ```

4. **Evaluation**: You can evaluate the generated images visually or using any evaluation metrics suitable for your application.

## Files Description
- `handwritten model using GAN.ipynb`: Utility functions for GAN Model.
- `handwritten model using GAN.ppt`: Describe briefly about the GAN model implemented in Handwriten using MNIST dataset
- `README.md`: This file.

## References

If you use this code in your research or project, please consider citing the following papers:

- Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
- Radford, Alec, et al. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

