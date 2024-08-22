import torch.nn as nn
from bingching import Transformer as Tf 
from bingching import BIOPHYSICS as bp 
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
import io
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.metrics import r2_score

df = pd.read_csv('radgyr_train.csv')

X = df['sequence'].tolist()
y = df['radius_of_gyration'].tolist()

model = Tf('z', bp, 768, 1024, 0.20, 512, 8, 8)
#def __init__(self, padding_token, biophysics, max_sl, d_model, dropout, ffn_hidden, num_heads, num_layers):

loss_fn = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

n_epochs = 5  # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)
 
# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:

            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            y_batch_tensor = y_batch.view(-1)  # Ensure y_batch is 1D

            print("started process " + str(int(start)))
            y_pred = model(X_batch).view(-1)  # Ensure y_pred is 1D
            print("finished process " + str(int(start)))
            
            # Loss calculation
            loss = loss_fn(y_pred, y_batch_tensor)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            bar.set_postfix(mse=float(loss))
    
    print("Doneeee!")
    
# Load the best model weights
model.load_state_dict(best_weights)

torch.save(model.state_dict(), 'model_weights.pth')

model.eval()
with torch.no_grad():

    y_pred = model(X_test).view(-1)  # Ensure y_pred is 1D
    y_test = y_test.view(-1)  # Ensure y_test is 1D
    
    mse = loss_fn(y_pred, y_test).item()
    
    # Calculate Adjusted R²
    r2 = r2_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
    adjusted_r2 = max(0, min(1, r2))
    
    # Calculate MAPE
    mape = torch.mean(torch.abs((y_test - y_pred) / y_test)).item()
    mape_score = 1 - mape
    
    print(f"Final Adjusted R²: {adjusted_r2}")
    print(f"Final MAPE Performance Score: {mape_score}")