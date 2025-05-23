import torch
import torch.nn as nn

class ALFAPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, temp=5):
        super().__init__()
        # Keep your existing network structure
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Add global context awareness
        self.global_context = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Final decision layer that considers both local and global features
        self.decision_layer = nn.Linear(hidden_dim // 2 + hidden_dim // 4, 1)
        # self.temperature = nn.Parameter(torch.tensor(2.0))
        self.temperature=temp

    def forward(self, x, y, x_global=None, y_global=None):
        # Process local features (same as before)
        combined = torch.cat((x, y), dim=1)
        local_features = self.feature_net(combined)
        
        # Process global context (if provided)
        if x_global is not None and y_global is not None:
            global_combined = torch.cat((x_global, y_global), dim=1)
            global_features = self.global_context(global_combined)
        else:
            # Use batch averages as simple global context if not provided
            # global_features = self.global_context(combined.mean(dim=0, keepdim=True).expand(combined.size(0), -1))
            print("x_global and y_global are None")
        
        # Combine local and global for final decision
        enhanced = torch.cat([local_features, global_features], dim=1)
        logits = self.decision_layer(enhanced)
        
        fusion_weight = torch.sigmoid(logits / self.temperature)

        return fusion_weight
    
