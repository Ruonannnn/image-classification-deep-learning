# Deep Learning for Plant leaf Disease Image Classification

This project builds and compares deep learning models (CNN and pretrained VGG16) for image classification in a real-world application: identifying plant species and diseases based on leaf images.

## Tools & Tech Stack

- **Framework**: PyTorch
- **Models**: CNN (custom), VGG16 (transfer learning)
- **Techniques**: Data augmentation, dropout, early stopping, hyperparameter tuning
- **Languages**: Python
- **Visualization**: Confusion matrix, loss/accuracy curves，data imbalance visualization

## Motivation

Traditional plant disease diagnosis relies on expert knowledge, which is often unavailable in rural or resource-limited regions. This project leverages deep learning to develop an automated, scalable, and accurate solution that can eventually support real-time agricultural diagnostics and further develop into mobile or web applications.

## Dataset

<p align="center">
  <img src="Plant leaves.png" width="600">
</p>

- **Source**: [Plant Village Dataset (Kaggle)](https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated/data)
- ~70,000 RGB images (256x256) across 9 plant types and multiple disease categories
- Renamed classes as "PlantName-DiseaseName" to form 29 distinct categories
- Merged and re-split data into balanced training, validation, and test sets
- Applied data augmentation (e.g., flipping, rotation, normalization) to improve generalization for CNN

## Model Architectures & Training Strategy

We built and compared two models to evaluate performance on the classification task:

- **Custom CNN**  
  A lightweight convolutional network trained from scratch.  
  - Architecture: 4 convolutional layers with ReLU and MaxPooling, followed by fully connected layers and softmax output  
  - Added Dropout to reduce overfitting, and applied EarlyStopping to capture the best validation epoch  
  - Hyperparameters: tuned learning rate, batch size, number of channels, and epoch count   

- **Pre-trained VGG16**  
  Fine-tuned from the VGG16 model pre-trained on ImageNet.  
  - Only the classifier layer was replaced and retrained; convolutional layers were frozen  
  - Removed data augmentation to avoid mismatch with ImageNet pretraining distribution  
  - Performed feature precomputation to reduce training time, and applied EarlyStopping to capture the best validation epoch
  - Hyperparameters: Tuned epoch count, learning rate, batch size, and dropout rate

## Modeling Summary

| Model     | Accuracy (Test) | Notes |
|-----------|-----------------|-------|
| CNN       | 97%             | Trained from scratch with custom architecture |
| VGG16     | 96%             | Fine-tuned last layers of pre-trained ImageNet model |

<p align="center">
  <img src="Plant leave.png" width="600">
</p>
<p align="center">
  <img src="Plant leave.png" width="600">
</p>
