from models.basic_nn_model import SimpleClassifier
from datetime import datetime

# Model config
glove_dim = 300
hidden_dim = 300
model_constructor = lambda e, f, g: SimpleClassifier(e, embed_dim=glove_dim, features_dim=g, hidden_dim=hidden_dim,
                                                     num_labels=2)

# Hyper-parameters
seed = 8008135
batch_size = 32
lr = 1e-4
weight_decay = 1e-3
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
save_path = f"../saves/basic_nn_{dt_string}/"

# Test configuration
path_punctuation = "C:/Users/Jelena/git_repos/irony-detection/saves/TAR_models/DNN/punct/best_model.pth"     # path to model with punctuation
path_no_punctuation = "C:/Users/Jelena/git_repos/irony-detection/saves/TAR_models/DNN/no_punct/best_model.pth"    # path to model with no punctuation
path_punctuation_features = "C:/Users/Jelena/git_repos/irony-detection/saves/TAR_models/DNN/punct_features/best_model.pth"     # path to model with punctuation and feat
#test_mode = "punctuation"  # can be punctuation or features
test_mode = "features"  # can be punctuation or features
exact = False
correction = True
