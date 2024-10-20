# README for Wonders of the World Classification Project

## Project Overview
This project aims to classify images of the wonders of the world using deep learning techniques, specifically through transfer learning with pre-trained models such as VGG16, ResNet50, and MobileNet. The goal is to leverage these models to achieve high accuracy in image classification tasks.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Installation
To run this project, ensure you have the following libraries installed:
```bash
pip install tensorflow numpy scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/ammaremad/wonders-of-the-world-classification.git
   cd wonders-of-the-world-classification
   ```

2. Prepare your dataset in the specified directory structure:
   ```
   Wonders of World/
       ├── Class1/
       │   ├── image1.jpg
       │   ├── image2.jpg
       │   └── ...
       ├── Class2/
       │   ├── image1.jpg
       │   ├── image2.jpg
       │   └── ...
       └── ...
   ```

3. Update the paths in the code to point to your dataset.

4. Run the training script:
   ```bash
   python train.py
   ```

## Data Preparation
The dataset consists of images of various wonders of the world, organized into separate folders for each class. The images are augmented using the `ImageDataGenerator` class from Keras, which applies transformations such as rotation, shifting, zooming, and flipping to enhance the training dataset.

## Model Training
The project implements three models using transfer learning:
- **VGG16**: Achieved a validation accuracy of 87.57% and a test accuracy of 93%.
- **ResNet50**: Underperformed with a validation accuracy of 12.30%.
- **MobileNet**: Achieved a validation accuracy of 85.47%.

### Training Process
- The models are trained using categorical cross-entropy loss and the Adam optimizer.
- Early stopping is implemented to prevent overfitting.
- Class weights are computed to handle any class imbalance in the dataset.

## Results
The VGG16 model was identified as the best performer, achieving:
- **Validation Accuracy**: 87.57%
- **Test Accuracy**: 93%

The results indicate that transfer learning can significantly enhance classification accuracy, especially with limited training data.

## Conclusion
This project successfully demonstrates the effectiveness of using pre-trained models for image classification tasks. Future work may involve fine-tuning the models and experimenting with additional architectures to further improve performance.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify any sections to better fit your project specifics or personal preferences!
