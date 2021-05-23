from models.cnn_dnn import CnnDnnClassifier

# Model config
model_constructor = lambda e, f, g: CnnDnnClassifier(e, embedding_dim=glove_dim, conv1_filters=conv1_filters,
                                               conv2_filters=conv2_filters, dropout_rate=dropout_rate,
                                               fc_neurons=fc_neurons, num_labels=num_labels, max_length=f,
                                               conv1_kernel=conv1_kernel, conv2_kernel=conv2_kernel,
                                               conv1_padding=conv1_padding, conv2_padding=conv2_padding)
batch_size = 32
lr = 1e-4
weight_decay = 1e-2
epochs = 200
early_stopping = True
early_stop_tolerance = 20

save_path = "../saves/cnn_dnn/"

# Task configuration
test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = False

remove_punctuation = True
glove_dim = 50
conv1_filters = 32
conv1_kernel = 5
conv1_padding = 2
conv2_filters = 64
conv2_kernel = 3
conv2_padding = 1
dropout_rate = 0.2
fc_neurons = 128
num_labels = 2
