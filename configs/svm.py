from sklearn.svm import SVC

# Default SVM config
backbone = SVC(
    kernel='rbf',
    tol=0.0001,
    C=1.0,
    verbose=0,
    random_state=None,
    max_iter=50000
)

test_task = 'A'
test_type = 'test'
test_emojis = True
test_irony_hashtags = False

remove_punctuation = True
use_features = True
