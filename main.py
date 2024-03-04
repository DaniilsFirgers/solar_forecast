import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from config.database import PRODUCTION_HISTORY_DB_NAME, PRODUCTION_HISTORY_COLLECTION_NAME
from database.main import mongo_handler
from data_handling.transform import CustomDataset
import matplotlib.pyplot as plt

SPLIT_RATIO = 0.8
BATCH_SIZE = 32
NUM_EPOCHS = 100
window = 24

filter_query = {
    "reference": "SADALES_TIKLS_lv_producer_43X-STJ02620895C_43Z-STO01766085R_12624502"
}

historical_data = mongo_handler.retrieve_data(
    PRODUCTION_HISTORY_DB_NAME, PRODUCTION_HISTORY_COLLECTION_NAME, filter_query)
if historical_data is None:
    print("Error retrieving data from MongoDB.")
    exit(1)

scaler = MinMaxScaler()
scaler.fit_transform(historical_data[["production"]])

# Prepare sequences for training
X, y = [], []
for i in range(len(historical_data) - window):
    X.append(historical_data.iloc[i:i + window].values)
    y.append(historical_data[i + window:i + window + 1]["production"].values)
X = np.stack(X)
y = np.stack(y)


# Split data into train and test sets
train_size = int(SPLIT_RATIO * len(X))
train_data, test_data = X[:train_size], X[train_size:]
print(f"Train data shape: {train_data}")
print(f"Test data shape: {test_data}")

# Create instances of your custom dataset
# train_dataset = CustomDataset(train_data)
# test_dataset = CustomDataset(test_data)

# Create instances of DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

model = nn.Sequential(
    # Adjust input size based on your sequence length
    nn.Linear(window, 64),
    nn.ReLU(),
    nn.Linear(64, 24)
).float()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, inputs in enumerate(train_loader):
        last_loss = 0.
        inputs = inputs.float()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # Assuming you are predicting consumption values
        targets = inputs
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % len(train_loader) == len(train_loader)-1:
            last_loss = running_loss / len(train_loader)  # loss per batch
            running_loss = 0.

    return last_loss


train_losses = []
val_losses = []


BEST_VAL_LOSS = 1_000_000_000
# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    # Train for one epoch
    train_loss = train_one_epoch()

    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, v_data in enumerate(test_loader):
            v_data = v_data.float()
            v_outputs = model(v_data)
            v_targets = v_data
            v_loss = criterion(v_outputs, v_targets)
            running_vloss += v_loss.item()

    val_loss = running_vloss / (i + 1)
    print('Epoch {} - LOSS train {} valid {}'.format(epoch, train_loss, val_loss))

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < BEST_VAL_LOSS:
        BEST_VAL_LOSS = val_loss
        torch.save(model.state_dict(), 'trained_models/model.pt')

plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()
