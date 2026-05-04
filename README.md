# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.
<img width="942" height="786" alt="WhatsApp Image 2026-05-04 at 1 53 36 PM" src="https://github.com/user-attachments/assets/16276bab-1a05-4b2c-a1f2-0a2e865cb6ca" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:MOHAMMED HAMZA M

### Register Number:212224230167

```python
# Name:
# Register Number:
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn. Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'Loss': []}


  def forward(self , x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3 (x)
    return x


 # Name:
# Register Number:
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['Loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

### Dataset Information
Include screenshot of the generated data
<img width="1919" height="1031" alt="image" src="https://github.com/user-attachments/assets/bd55ca56-e06d-4c03-bce9-07cf31d1a07e" />

### OUTPUT
<img width="1919" height="1078" alt="image" src="https://github.com/user-attachments/assets/55b35682-28ae-4fd0-b8a1-09e9afedeca1" />

### Training Loss Vs Iteration Plot
<img width="1919" height="1078" alt="image" src="https://github.com/user-attachments/assets/55b35682-28ae-4fd0-b8a1-09e9afedeca1" />

### New Sample Data Prediction
Include your sample input and output here
<img width="1919" height="1031" alt="image" src="https://github.com/user-attachments/assets/a82e8920-0400-4244-a920-045826c22218" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
