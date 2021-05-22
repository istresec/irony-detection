from models.simple_rnn import RNNClassifier

# Model config
model = RNNClassifier
batch_size = 32
lr = 1e-4
weight_decay = 1e-2
epochs = 200
early_stopping = True
early_stop_tolerance = 20

save_path = "../saves/simple_rnn/"

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = False

remove_punctuation = True
glove_dim = 300
hidden_dim = 300
num_layers = 2
dropout = 0.25
