from models.basic_nn_model import SimpleClassifier

# Model config
model_constructor = lambda e, f, g: SimpleClassifier(e, embed_dim=glove_dim, features_dim=g, hidden_dim=hidden_dim,
                                                     num_labels=2)
batch_size = 32
lr = 1e-3
weight_decay = 1e-3
epochs = 200
early_stopping = True
early_stop_tolerance = 20

save_path = "../saves/basic_nn/"

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = False

remove_punctuation = False
glove_dim = 50
hidden_dim = 50
num_layers = 2
dropout = 0.25
