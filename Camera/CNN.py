import pandas as pd 
import torch.nn as nn

# Load CNN model
class CNN(nn.Module):
    def __init__(self,K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(

                        # conv1
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

                        # conv2
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
                        # conv3
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

                        # conv4
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176,1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)

        # Flatten
        out = out.view(-1, 50176)

        # Fully connected
        out = self.dense_layers(out)

        return out        
    
idx_to_classes = {0: 'scab',
                  1: 'Black_rot',
                  2: 'Cedar_rust',
                  3: 'healthy',
                  4: 'Background_without_leaves',
                  5: 'healthy',
                  6: 'Powdery_mildew',
                  7: 'healthy',
                  8: 'Cercospora_leaf_spot Gray_leaf_spot',
                  9: 'Common_rust',
                  10: 'Northern_Leaf_Blight',
                  11: 'healthy',
                  12: 'Black_rot',
                  13: 'Esca_(Black_Measles)',
                  14: 'Leaf_blight_(Isariopsis_Leaf_Spot)',
                  15: 'healthy',
                  16: 'Haunglongbing_(Citrus_greening)',
                  17: 'Bacterial_spot',
                  18: 'healthy',
                  19: 'Bacterial_spot',
                  20: 'healthy',
                  21: 'Early_blight',
                  22: 'Late_blight',
                  23: 'healthy',
                  24: 'healthy',
                  25: 'healthy',
                  26: 'Powdery_mildew',
                  27: 'Leaf_scorch',
                  28: 'healthy',
                  29: 'Bacterial_spot',
                  30: 'Early_blight',
                  31: 'Late_blight',
                  32: 'Leaf_Mold',
                  33: 'Septoria_leaf_spot',
                  34: 'Spider_mites Two-spotted_spider_mite',
                  35: 'Target_Spot',
                  36: 'Yellow_Leaf_Curl_Virus',
                  37: 'mosaic_virus',
                  38: 'healthy'
                  }