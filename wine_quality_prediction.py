import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import model_evaluation


df = pd.read_csv(r'C:\Users\Wiktoria\Desktop\Python Basics\Project4\wine_quality_data.csv')

df.isna().sum()

df

pd.set_option('display.max_columns', None)
df.describe()

df['quality'].value_counts()

plt.figure()
df['quality'].value_counts().plot(kind='bar')
plt.xlabel('Quality')
plt.ylabel('Counts')
plt.title('Value Counts By Wine Quality')

#removal of classes 3 and 8 - very low number of records compared to the others
class_to_drop = df[(df['quality'] == 3) | (df['quality'] == 8)].index

df = df.drop(class_to_drop)

X = df.drop(columns='quality')

y = df['quality']

y.value_counts()

#oversampling on class 4
sampling_strategy = {4:200}

oversampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)

X, y = oversampler.fit_resample(X, y)

y.value_counts()

#undersampling for classes 5 and 6
undersampling_strategy = {5: 200, 6:200}

undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy,
                                  random_state=42)

X, y = undersampler.fit_resample(X, y)

y.value_counts() #200, 200, 200, 199

df_resampled = X
df_resampled['quality'] = y

df_resampled.corr()

corr = df_resampled.corr()

plt.figure()
sns.heatmap(corr, annot=True)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

#logistic regression
modelLR = LogisticRegression()

modelLR.fit(X_train_scaled, y_train)

#preds = modelLR.predict(X_test_scaled)

model_evaluation.evaluate(modelLR, X_train_scaled, 
                          y_train,'Logistic Regression - Train')
model_evaluation.evaluate(modelLR, X_test_scaled, 
                          y_test,'Logistic Regression - Test')

#label range shift
y_train_shifted = y_train - 4
y_test_shifted = y_test - 4

#labels conversion to one-hot vectors
y_train_one_hot = to_categorical(y_train_shifted)
y_test_one_hot = to_categorical(y_test_shifted)

#check the shape of labels
print(y_train_one_hot.shape)
print(y_test_one_hot.shape)


#neural network
#input layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5, activation='relu', input_shape = (X_train_scaled.shape[1], )))
#hidden layers
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='relu'))
#output layer
model.add(tf.keras.layers.Dense(4, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#model training
history = model.fit(X_train_scaled, y_train_one_hot, validation_data=(X_test_scaled, y_test_one_hot), epochs=100, verbose=1)


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['taining', 'validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve during model training')


plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['taining', 'validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy curve during model training')