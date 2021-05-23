from models.lstm_rnn import RNNClassifier

# Model config
model_constructor = lambda e, f: RNNClassifier(e, embed_dim=glove_dim, hidden_dim=hidden_dim,
                                               num_layers=num_layers, dropout=dropout, max_len=f)
batch_size = 32
lr = 1e-5
weight_decay = 1e-3
epochs = 200
early_stopping = True
early_stop_tolerance = 20

save_path = "../saves/lstm_rnn/"

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = False

remove_punctuation = True
glove_dim = 50
hidden_dim = 50
num_layers = 5
dropout = 0.25
