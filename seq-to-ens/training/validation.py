import torch 
from bingching import Transformer as Tf, BIOPHYSICS as bp
import pandas as pd
import torch.nn as nn
import torch
import pandas as pd 
from sklearn.metrics import r2_score

#def __init__(self, padding_token, biophysics, max_sl, d_model, dropout, ffn_hidden, num_heads, num_layers):
model = Tf('z', bp, 768, 1024, 0.20, 512, 8, 12)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

df = pd.read_csv('radgyr_val.csv')
X_new = df['sequence'].tolist()
y_new = df['radius_of_gyration'].tolist()

# Ensure the model is in evaluation mode
model.eval()

# Make predictions on the new dataset
with torch.no_grad():
    y_pred_new = (model(X_new)).squeeze()
    y_new_tensor = torch.tensor(y_new, dtype=torch.float32)
    y_new_tensor.squeeze()

# Evaluate the model's performance
r2 = r2_score(y_new_tensor.cpu().numpy(), y_pred_new.cpu().numpy())

print(f"R² on new dataset: {r2}")

