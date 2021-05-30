from models.cnn_lstm import CnnRnnClassifier

# Model config
glove_dim = 300
hidden_dim = 300
conv1_filters = 32
conv1_kernel = 5
conv1_padding = 2
conv2_filters = 64
conv2_kernel = 3
conv2_padding = 1
dropout_rate = 0.2
fc_neurons = 128
num_labels = 2
model_constructor = lambda e, f, g: CnnRnnClassifier(e, embedding_dim=glove_dim, conv1_filters=conv1_filters,
                                               conv2_filters=conv2_filters, dropout_rate=dropout_rate,
                                               lstm_hidden_size=hidden_dim, fc_neurons=fc_neurons,
                                               num_labels=num_labels, conv1_kernel=conv1_kernel,
                                               conv1_padding=conv1_padding, conv2_kernel=conv2_kernel,
                                               conv2_padding=conv2_padding, max_length=f, features_dim=g)

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
save_path = f"../saves/cnn_lstm_{seed}_{punctuation}_{features}/"
