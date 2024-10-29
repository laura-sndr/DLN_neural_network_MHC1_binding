import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

df = "/path/to/your/peptideMHCI_binding.csv"

class PeptideClass(Dataset):
    
    # There, we store important information such as labels and the list of IDs that we wish to generate at each pass.
    def __init__(self, df):
        self.dataframe = pd.read_csv(df)
        self.dataframe = self.dataframe.loc[self.dataframe.binding != 5]
        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_index = {aa: i for i, aa in enumerate(self.alphabet)} 
            
    def __len__(self):
        return len(self.dataframe)
    
    def encode_sequence(self, peptide):
        indices = torch.tensor([self.aa_index[aa] for aa in peptide])           # peptide sequence is converted to a tensor with using the aa_index dictionary
        one_hot_encoded = F.one_hot(indices, num_classes=20)                    # converts into one_hot_encoded vectors with 20 classes (amino acids)
        return one_hot_encoded.float()
    
    def __getitem__(self, idx): # second argument = index of list
        # select sample
        x = torch.reshape((self.encode_sequence(self.dataframe.iloc[idx].peptide)), (-1,))  # (-1,) here means the shape will be found by itself (all the elements will be put into one row)
        y = torch.tensor([self.dataframe.iloc[idx].binding]).float()
        return x, y

print(len(df))

peptides = PeptideClass(df)
print(peptides[234])

# DataSplit
train_set, val_set, test_set = torch.utils.data.random_split(peptides, [0.7, 0.2, 0.1])
print(len(train_set))
print(len(val_set))
print(len(test_set))

# DataLoader

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)               # shuffle needed so that it's different every epoch
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)

# neural network class

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(180,120)
        self.fc2 = nn.Linear(120,60)
        self.fc3 = nn.Linear(60,1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
        
net = Network()

print(net(peptides[0][0]))


# Loss function and optimizer
criterion = nn.BCELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
optimizer = optim.Adam(net.parameters(), lr=0.001) 

# Training loop

for epoch in range(10):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):                                  # Iterate over the training data
        inputs, labels = data                                                   # Get the inputs and labels

        optimizer.zero_grad()                                                   # Zero the parameter gradients

        # Forward pass
        outputs = net(inputs)                                                   # Get the network outputs
        loss = criterion(outputs, labels)
        loss.backward()                                                         # Backward pass (compute gradients)
        optimizer.step()  
        
        if i % 10 == 0 :
            print(f"Training loss: {loss.item()}")

    # Validation loop
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = net(inputs).round()
            total += labels.size(0)                                            
            correct += (outputs == labels).sum().item()  
            
    print(f"% of correctly predicted entries (validation): {correct/total * 100}")
    
# Test set
test_x, test_y = next(iter(test_loader))  
outputs = net(test_x).round()
total = test_y.size(0)
correct = (outputs == test_y).sum().item()
print(f"% of correctly predicted entried (test): {correct/total * 100}")
print(test_y.shape)


            
        
        
        
        
