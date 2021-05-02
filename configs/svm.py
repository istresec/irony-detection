from sklearn.svm import LinearSVC

# Default SVM config
backbone = LinearSVC(
    penalty='l2',
    loss='squared_hinge',
    dual=True,
    tol=0.0001,
    C=1.0,
    multi_class='ovr',
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    verbose=0,
    random_state=None,
    max_iter=1000
)

test_task = 'A'
test_type = 'train'
test_emojis = True
test_irony_hashtags = False

remove_punctuation = True
