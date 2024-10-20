
# Digit Recognition Using Convolutional Neural Network (CNN)

This project demonstrates the use of a Convolutional Neural Network (CNN) for recognizing handwritten digits from the popular MNIST dataset. The model is built using PyTorch and trained on the MNIST dataset to classify images of digits (0-9).

## Dataset

The dataset used is the MNIST dataset, which contains 70,000 images of handwritten digits (28x28 pixels in grayscale). The dataset is split into:
- **60,000 training images**
- **10,000 test images**

Each image is labeled with the corresponding digit (0-9).

## Project Workflow

The project workflow in the notebook `Digit_Recognition.ipynb` includes the following steps:

1. **Data Loading**: The MNIST dataset is loaded using the `torchvision` library. 
2. **CNN Model Definition**: A CNN is defined using PyTorch, which includes convolutional layers, dropout, and fully connected layers.
3. **Training**: The CNN model is trained for 10 epochs using the Adam optimizer and cross-entropy loss.
4. **Testing**: The model's performance is evaluated on the test dataset, and accuracy is calculated.
5. **Prediction & Visualization**: A sample image from the test set is used to make a prediction, and the image along with the predicted digit is displayed.

### Model Architecture

The architecture of the CNN used in this project includes:
- **Two convolutional layers**: Extract features from the input images.
- **Max pooling layers**: Reduce the dimensionality of the feature maps.
- **Dropout**: Added to prevent overfitting.
- **Fully connected layers**: Used to map the extracted features to the output classes (0-9).
- **ReLU activation**: Used in between layers to introduce non-linearity.
- **Softmax**: To obtain the final probability distribution for digit classes.

### Code Example

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
```

### Training Loop

The model is trained using the Adam optimizer with cross-entropy loss. Below is an example of how the model is trained and tested:

```python
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} '
                  f'({100. * batch_idx / len(loaders["train"]):.0f}%)]	Loss: {loss.item():.6f}')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loaders['test'].dataset)
    print(f'
Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)} '
          f'({100. * correct / len(loaders["test"].dataset):.0f}%)')
```

### Visualization

After training, the model is used to predict the digit of a sample image from the test set, and the predicted digit along with the image is visualized.

```python
model.eval()
data, target = test_data[3]
data = data.unsqueeze(0).to(device)
output = model(data)
prediction = output.argmax(dim=1, keepdim=True).item()
print(f'Prediction: {prediction}')
image = data.squeeze(0).squeeze(0).cpu().numpy()
plt.imshow(image, cmap='gray')
plt.show()
```

## Requirements

To run this project, the following Python libraries are required:

- `torch`
- `torchvision`
- `matplotlib`
- `tensorflow` (for data processing)

## Running the Notebook

This project can be run on Google Colab or any local Jupyter Notebook environment. You can directly access the notebook from [Google Colab](https://colab.research.google.com/drive/1QS3TDQuav6K_j9_q4xeOQeap36xbXfnN).

## Conclusion

This project provided an introduction to using Convolutional Neural Networks for digit recognition. By following this guide, I trained a CNN on the MNIST dataset and predicted handwritten digits.

## Acknowledgments

- **MNIST Dataset**: The dataset was sourced from the [MNIST database](http://yann.lecun.com/exdb/mnist/).
