import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found")

# Hyperparameters
num_epochs = 33  # Number of training epochs
learning_rate = 0.1  # Learning rate for the optimizer

# Data preprocessing and augmentation for training set
transform_train = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),  # Normalize with CIFAR-10 mean and std
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for augmentation
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # Randomly crop the image with padding
])

# Data preprocessing for test set (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
])

# Load CIFAR-10 training data
trainset = torchvision.datasets.CIFAR10(
    root='cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)  # DataLoader for batching and shuffling

# Load CIFAR-10 test data
testset = torchvision.datasets.CIFAR10(
    root='cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)  # DataLoader for batching

# BasicBlock class defines a single block in the ResNet architecture
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # Shortcut connection to downsample the input when the dimensions do not match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        # Forward pass through the block
        out = F.relu(self.bn1(self.conv1(x)))  # First conv + batch norm + ReLU
        out = self.bn2(self.conv2(out))  # Second conv + batch norm
        out += self.shortcut(x)  # Add the shortcut connection
        out = F.relu(out)  # Apply ReLU
        return out

# ResNet model class
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # Initial convolutional layer and batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Layers 1 to 4 in ResNet, each containing multiple blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Final fully connected layer for classification
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        # Create a layer of `num_blocks` blocks with specified number of planes and stride
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Forward pass through the network
        out = F.relu(self.bn1(self.conv1(x)))  # Initial conv + batch norm + ReLU
        out = self.layer1(out)  # Layer 1
        out = self.layer2(out)  # Layer 2
        out = self.layer3(out)  # Layer 3
        out = self.layer4(out)  # Layer 4
        out = F.avg_pool2d(out, 4)  # Average pooling
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.linear(out)  # Fully connected layer
        return out

# Function to create a ResNet-18 model
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# Create the model, move it to the GPU (if available), and print model info
model = ResNet18()
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)  # Stochastic Gradient Descent (SGD) optimizer

# Learning rate scheduling
total_step = len(train_loader)
sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr=learning_rate, 
                                                   step_size_up=15, step_size_down=15, mode='triangular', 
                                                   verbose=False)  # Cyclic learning rate scheduler
sched_linear_3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.005/learning_rate , end_factor=0.005/5, total_iters=30, verbose=False)  # Linear learning rate scheduler
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30])  # Combine schedulers

# Training loop
model.train()  # Set model to training mode
print("> Training")
start = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # Move images to device
        labels = labels.to(device)  # Move labels to device
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print training progress every 100 steps
        if (i+1) % 100 == 0 :
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    # Step the scheduler at the end of each epoch
    scheduler.step()
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Testing loop
print("> Testing")
start = time.time()
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient computation for testing
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate and print the test accuracy
    accuracy = 100 * correct / total
    print("Test Accuracy: {:.2f} %".format(accuracy))

end = time.time()
elapsed = end - start
print("Testing took {:.2f} secs or {:.2f} mins in total".format(elapsed, elapsed/60))
