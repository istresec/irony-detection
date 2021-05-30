from models.simple_rnn import RNNClassifier

# Model config
model_constructor = lambda e, f, g: RNNClassifier(e, embed_dim=glove_dim, hidden_dim=hidden_dim,
                                                  num_layers=num_layers, dropout=dropout)
batch_size = 32
lr = 1e-4
weight_decay = 5e-4
epochs = 200
early_stopping = True
early_stop_tolerance = 30

save_path = "../saves/simple_rnn/"

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = False

remove_punctuation = False
use_features = False
glove_dim = 50
hidden_dim = 50
num_layers = 2
dropout = 0.25
