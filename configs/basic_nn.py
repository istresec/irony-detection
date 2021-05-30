from models.basic_nn_model import SimpleClassifier

# Model config
glove_dim = 300
hidden_dim = 300
model_constructor = lambda e, f, g: SimpleClassifier(e, embed_dim=glove_dim, features_dim=g, hidden_dim=hidden_dim,
                                                     num_labels=2)

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
use_features = True

# Save configuration
punctuation = "punctuation" if not remove_punctuation else "removed"
features = "features" if use_features else ""
save_path = f"../saves/basic_nn_{seed}_{punctuation}_{features}/"
