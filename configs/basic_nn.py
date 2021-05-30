from models.basic_nn_model import SimpleClassifier

# Model config
model_constructor = lambda e, f, g: SimpleClassifier(e, embed_dim=glove_dim, features_dim=g, hidden_dim=hidden_dim,
                                                     num_labels=2)
batch_size = 32
lr = 1e-4
weight_decay = 1e-3
epochs = 200
early_stopping = True
early_stop_tolerance = 20

save_path = "../saves/basic_nn_removed/"

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = True

remove_punctuation = False
use_features = True
glove_dim = 300
hidden_dim = 300
