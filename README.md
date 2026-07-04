# Effects of Hyperparameters on EfficientNet-B0 and Food-11 Dataset

This project aims to analyze the effects of key hyperparameters on the performance of deep neural networks. The components covered in this repository include optimization techniques (learning rate, learning rate schedule) and regularization techniques (weight decay, mixup data augmentation), with iterative ablation experiments performed to identify optimal hyperparameter values.

EfficientNet-B0 is a lightweight convolutional neural network architecture that is widely used for image classification tasks. The Food-11 dataset is a large collection of 16,643 food images spanning 11 categories, making it a suitable benchmark for evaluating the effects of hyperparameters on a real-world image classification problem. Through iterative ablation testing on this network and dataset, the optimal set of hyperparameters is identified and used to generate predictions on unseen test data, achieving an accuracy of **81.63%**.

## Tech Stack Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-blue?style=for-the-badge)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

## Methodology

1. **Data Preprocessing**  
   The Food-11 dataset consists of 16,643 images which are roughly divided in the ratio of 6:2:2 for training (9,866), validation (3,430) and test (3,347) datasets respectively. In the data pre-processing pipeline, the images are first resized to a resolution of 100 x 100. Following this, the mean and standard deviation of each color channel in the training set is computed and used for normalizing the data. The resizing and normalizing is applied to both the training and validation set. Additionally, data augmentation techniques such as Random Horizontal Flip (with a probability factor of 0.5) and Random Cropping with Padding (zero-padding the image to 124 x 124 and randomly cropping back to 100 x 100) are applied only to the training set to increase the diversity of the training data.

2. **Learning Rate Ablation**  
   Learning rate is an optimization hyperparameter that determines the step size taken to update the model’s weights, essentially describing the stability and speed of convergence during training. To test the effects of learning rate, the model was trained for 15 epochs at a batch size of 128 with three different learning rates (0.1, 0.025, 0.001). The observations showed that a low learning rate (0.001) performs slow convergence, while a high learning rate (0.1) is highly unstable, exhibiting a "zig-zag" like pattern in validation accuracy. The optimal learning rate was identified as **0.025**, achieving 17.77% more training accuracy and 10.79% more validation accuracy in comparison to the other tested values.

3. **Learning Rate Schedule Ablation**  
   During training, the learning rate should be reduced after a certain amount of epochs such that convergence does not plateau. In this stage, cosine annealing is compared against a constant learning rate schedule. Cosine annealing reduces the learning rate in the form of a cosine curve, starting with a high initial learning rate and gradually decreasing it. It was observed that cosine annealing improves the validation accuracy by **2.8%** in comparison to a constant learning rate. The gradual decrease of learning rate through cosine annealing helps reduce the gradient noise and allows the weights to converge to local minima in a more stable and smoother fashion.

4. **Weight Decay Ablation**  
   Weight decay is a regularization method used in neural networks that helps prevent overfitting by penalizing large weights, functionally similar to L2 regularization used in Ridge Regression. To evaluate this regularization effect, the training pipeline was executed for 300 epochs and the model's behavior was compared using weight decay coefficients λ = 5 × 10⁻⁴ and λ = 1 × 10⁻⁴. While both coefficients performed near-identically on the training set, a higher weight decay value (0.0005) achieved a validation accuracy of **78.43%** in comparison to 74.81% achieved by the smaller weight decay value. This 3.62% improvement showed that a higher weight decay coefficient more effectively suppresses the weights of the model, thereby reducing model complexity and preventing overfitting.

5. **Mixup Data Augmentation Ablation**  
   Mixup is a data augmentation technique that creates new image samples through combination of two image and label pairs. The hyperparameter α dictates the beta distribution through which the mixing factor λ is sampled. A small α value (0.2) indicates that λ values lie near 0 or 1, meaning the new sample is often very close to one of the original sample images. With the incorporation of Mixup, the training accuracy dropped from 99.81% to 91.22%, indicating that training becomes more difficult. However, this regularization technique introduced the highest validation accuracy (**79.83%**) and the lowest loss (0.6834) observed so far, showing that Mixup helps improve the generalization and predictive capabilities of the model.

6. **Final Evaluation**  
   For the evaluation on the test dataset, the most optimal hyperparameters across the experimental study were chosen, including a learning rate of 0.025, cosine annealing learning rate scheduler, weight decay of 0.0005, and Mixup data augmentation. The performance of the EfficientNet-B0 network trained on the Food-11 dataset with these configurations for 300 epochs was evaluated. The final evaluation on the unseen test set yielded an accuracy of **81.63%** and a loss of 0.6329. This performance exceeds the metrics achieved on the validation dataset, confirming that the chosen optimal hyperparameters resulted in a well-generalized model.

## Key Results

| Experiment | Best Configuration | Validation Accuracy |
|------------|-------------------|---------------------|
| Learning Rate | 0.025 | 52.92% |
| Learning Rate Schedule | Cosine Annealing | 75.13% |
| Weight Decay | 0.0005 | 78.43% |
| Mixup Data Augmentation | α = 0.2 | 79.83% |
| **Final Test Set Accuracy** | **All optimal combined** | **81.63%** |

## Future Scope

The experimental study can further be extended by exploring additional hyperparameters such as different optimizers (Adam, AdamW), varying batch sizes, and alternative learning rate schedulers like step decay or warm restarts. Furthermore, the study can be extended to evaluate the effects of these hyperparameters on other lightweight architectures like MobileNet or ShuffleNet to draw more generalized conclusions on hyperparameter tuning for image classification tasks.

## Academic Context
Completed during my Graduate studies at **Nanyang Technological University (NTU), Singapore**.

* **Course:** AI6103 - Deep Learning & Applications
* **Semester:** AY 2025/2026, Semester 2

## Usage
**Note for current/future NTU students:** While this repository is public, please ensure you adhere to NTU's Academic Integrity Policy. This is intended as a reference for my personal portfolio; using this code for your own graded assignments is strictly prohibited by the University.
