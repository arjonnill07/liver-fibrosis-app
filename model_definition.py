import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Define the custom model with attention mechanism
class AttentionTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_classes):
        super(AttentionTransformer, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)  # Updated line
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, 1024)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        features = self.efficientnet(x)  # Output shape: (batch_size, 1024)
        features = features.unsqueeze(1)  # Add seq_len dimension (1)
        attention_output, _ = self.attention(features, features, features)
        output = self.fc(attention_output.squeeze(1))  # Remove seq_len dimension
        return output
    
# --- END OF MODEL DEFINITION ---

# Function to load the model (This part is okay)
def load_model(model_path, num_classes=5, input_dim=1024, num_heads=4): # Defaults match notebook
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the CORRECT model class
    model = AttentionTransformer(input_dim=input_dim, num_heads=num_heads, num_classes=num_classes)
    try:
        # Load the state dict - use map_location for CPU deployment if trained on GPU
        # Use strict=True first, as the definitions should now match perfectly
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    except RuntimeError as e:
        print(f"Error loading state_dict with strict=True: {e}")
        print("This might indicate a subtle difference between saved and current model def.")
        print("Attempting to load with strict=False as a fallback.")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            print("Loaded with strict=False. Check if all expected layers were loaded.")
        except Exception as e2:
             print(f"Loading with strict=False also failed: {e2}")
             raise RuntimeError(f"Could not load the model weights from {model_path}") from e2

    model.to(device)
    model.eval() # Set to evaluation mode!
    return model, device 