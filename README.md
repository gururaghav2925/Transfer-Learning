# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.


## Problem Statement and Dataset
The experiment aims to develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron. The model will be trained on a dataset containing images of various defected and non-defected capacitors to enhance defect detection accuracy. Optimization techniques will be applied to improve performance, and the model will be evaluated to ensure reliable classification for capacitor quality assessment in manufacturing.


## DESIGN STEPS

### STEP 1: Problem Statement  
Define the objective of distinguishing between defected and non-defected capacitors using a binary classification model based on a pretrained VGG19.  

### STEP 2: Dataset Collection  
Use a dataset containing images of defected and non-defected capacitors for model training and evaluation.  

### STEP 3: Data Preprocessing  
Resize images to match VGG19 input dimensions, normalize pixel values, and create DataLoaders for efficient batch processing.  

### STEP 4: Model Architecture  
Modify the pretrained VGG19 by replacing the last layer with a single neuron using a sigmoid activation function for binary classification.  

### STEP 5: Model Training  
Train the model using a suitable loss function (Binary Cross-Entropy) and optimizer (Adam) for multiple epochs to enhance defect detection accuracy.  

### STEP 6: Model Evaluation  
Evaluate the model on unseen data using accuracy, precision, recall, and an ROC curve to assess classification performance.  

### STEP 7: Model Deployment & Visualization  
Save the trained model, visualize predictions, and integrate it into a manufacturing quality assessment system if needed.

## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning

from torchvision.models import VGG19_Weights
model = models.vgg19(weights=VGG19_Weights.DEFAULT)

```


# Modify the final fully connected layer to match the dataset classes
```python
num_classes = len(train_dataset.classes)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features,1)
```
# Include the Loss function and optimizer
```python
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)
```



# Train the model
```python
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float() )
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Guru Raghav Ponjeevith V")
    print("Register Number:212223220027")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```




## OUTPUT
### Training Loss
</br>
</br>
</br>

![alt text](image.png)

###  Validation Loss Vs Iteration Plot

![alt text](image-1.png)

### Confusion Matrix

![alt text](image-2.png)

</br>
</br>
</br>

### Classification Report

![alt text](image-3.png)
</br>
</br>
</br>

### New Sample Prediction
```
predict_image(model, image_index=65, dataset=test_dataset)
```
![alt text](image-4.png)


```

predict_image(model, image_index=95, dataset=test_dataset)
```
![alt text](image-5.png)
</br>
</br>
</br>

## RESULT
Thus, the Transfer Learning for classification using the VGG-19 architecture has been successfully implemented.
</br>
</br>
</br>
