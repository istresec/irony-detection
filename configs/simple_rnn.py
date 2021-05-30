from models.simple_rnn import RNNClassifier

# Model config
glove_dim = 50
hidden_dim = 50
num_layers = 2
dropout = 0.25
model_constructor = lambda e, f, g: RNNClassifier(e, embed_dim=glove_dim, hidden_dim=hidden_dim,
                                                  num_layers=num_layers, dropout=dropout, features_dim=g)

# Hyper-parameters
seed = 8008135
batch_size = 32
lr = 1e-4
weight_decay = 5e-4
epochs = 1
early_stopping = True
early_stop_tolerance = 50

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = True

remove_punctuation = False
use_features = True

# Save configuration
punctuation = "punctuation" if not remove_punctuation else "removed"
features = "features" if use_features else ""
save_path = f"../saves/simple_rnn_{seed}_{punctuation}_{features}/"

