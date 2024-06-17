import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_network(model, X_train_scaled, X_test_scaled, y_train_one_hot, y_test_one_hot):

    #restoring original labels (to fit 4-7)
    y_train_labels = np.argmax(y_train_one_hot, axis=1) + 4
    
    y_test_labels = np.argmax(y_test_one_hot, axis=1) + 4
    
    model_preds_train = model.predict(X_train_scaled)
    model_preds_test = model.predict(X_test_scaled)
    
    if model_preds_train.ndim > 1:
        model_preds_train_labels = np.argmax(model_preds_train, axis=1)
    else:
        model_preds_train_labels = model_preds_train
    
    if model_preds_test.ndim > 1:
        model_preds_test_labels = np.argmax(model_preds_test, axis=1)
    else:
        model_preds_test_labels = model_preds_test
        
    #adding a shift to predictions
    model_preds_train_labels += 4
    model_preds_test_labels += 4
    
    #train
    conf_matrix_train = confusion_matrix(y_train_labels, model_preds_train_labels)
    print("Confusion matrix for training set")
    print(conf_matrix_train)
    
    print('Classification report for training set')
    print(classification_report(y_train_labels, model_preds_train_labels))
    
    accuracy_train = accuracy_score(y_train_labels, model_preds_train_labels)
    print('Accuracy for training set:', accuracy_train)
    
    
    #test
    conf_matrix_test = confusion_matrix(y_test_labels, model_preds_test_labels)
    print("Confusion matrix for test set")
    print(conf_matrix_test)
    
    print('Classification report for test set')
    print(classification_report(y_test_labels, model_preds_test_labels))
    
    accuracy_test = accuracy_score(y_test_labels, model_preds_test_labels)
    print('Accuracy for test set:', accuracy_test)