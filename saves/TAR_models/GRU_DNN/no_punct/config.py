from models.simple_rnn import RNNClassifier
from datetime import datetime

# Model config
glove_dim = 300
hidden_dim = 300
num_layers = 2
dropout = 0.2
model_constructor = lambda e, f, g: RNNClassifier(e, embed_dim=glove_dim, hidden_dim=hidden_dim,
                                                  num_layers=num_layers, dropout=dropout, features_dim=g)

# Hyper-parameters
seed = 8008135
batch_size = 32
lr = 1e-4
weight_decay = 2e-4
epochs = 1000
early_stopping = True
early_stop_tolerance = 50

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = True

remove_punctuation = True
use_features = False

# Save configuration
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
save_path = f"../saves/simple_rnn_{dt_string}/"

# Test configuration
path_first_model = "../saves/probni/b.pth"     # path to model with punctuation
path_second_model = "../saves/probni/a.pth"    # path to model with no punctuation or with features
test_mode = "punctuation"  # can be punctuation or features
exact = False
correction = True
