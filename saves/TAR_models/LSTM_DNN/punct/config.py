from models.lstm_rnn import RNNClassifier
from datetime import datetime

# Model config
glove_dim = 300
hidden_dim = 50
num_layers = 2
dropout = 0.2
model_constructor = lambda e, f, g: RNNClassifier(e, embed_dim=glove_dim, hidden_dim=hidden_dim,
                                               num_layers=num_layers, dropout=dropout, max_len=f, features_dim=g)

# Hyper-parameters
seed = 8008135
batch_size = 32
lr = 1e-4
weight_decay = 5e-4
epochs = 1000
early_stopping = True
early_stop_tolerance = 50

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = True

remove_punctuation = False
use_features = False

# Save configuration
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
save_path = f"../saves/lstm_rnn_{dt_string}/"

# Test configuration
path_punctuation = "/saves/TAR_models/LSTM_DNN/punct/best_model.pth"  # path to model with punctuation
path_no_punctuation = "/saves/TAR_models/LSTM_DNN/no_punct/best_model.pth"  # path to model with no punctuation
path_punctuation_features = "/saves/TAR_models/LSTM_DNN/punct_features/best_model.pth"  # path to model with punctuation and feat
test_mode = "punctuation"  # can be punctuation or features
#test_mode = "features"  # can be punctuation or features
exact = False
correction = True
