import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier


def read_from_db(database_name='wine_quality_database', 
                 collection_name='wine_quality'):
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]
    
    data = list(collection.find())
    
    df = pd.DataFrame(data)
    
    return df

df = read_from_db()

df.head()

df = df.drop(columns=['_id'])

X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

#label range shift
y_train_shifted = y_train - 4
y_test_shifted = y_test - 4

#labels conversion to one-hot vectors
y_train_one_hot = to_categorical(y_train_shifted)
y_test_one_hot = to_categorical(y_test_shifted)


def create_model():
    #input layer
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5, activation='relu', input_shape = (X_train_scaled.shape[1], )))
    #hidden layers
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='relu'))
    #output layer
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

#packaging the model in Sci-Keras
keras_classifier = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=1)

#definition of KFold
kf = KFold(n_splits=5 , shuffle=True , random_state=42)

#cross validation
scores = cross_val_score(keras_classifier, X_train_scaled, y_train_one_hot, cv=kf,
                         scoring='accuracy')


print(f'Mean classification accuracy for neural network model: {scores.mean()}')