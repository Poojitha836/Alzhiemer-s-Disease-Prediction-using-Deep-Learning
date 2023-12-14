import numpy as np 
import pandas as pd 
import seaborn as sns 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import os 
from distutils.dir_util import copy_tree, remove_tree 
from PIL import Image 
from random import randint 
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import matthews_corrcoef as MCC 
from sklearn.metrics import balanced_accuracy_score as BAS 
from sklearn.metrics import classification_report, confusion_matrix 
import tensorflow_addons as tfa 
from keras.utils.vis_utils import plot_model 
from tensorflow.keras import Sequential, Input 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.layers import Conv2D, Flatten 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.applications.inception_v3 import 
InceptionV3 
from tensorflow.keras.preprocessing.image import 
ImageDataGenerator as IDG 
from tensorflow.keras.layers import SeparableConv2D, 
BatchNormalization, MaxPool2D 
print("TensorFlow Version:", tf._version_) 
base_dir = "/kaggle/input/alzheimers-dataset-4-class-ofimages/Alzheimer_s Dataset/" 
root_dir = "./" 
test_dir = base_dir + "test/" 
train_dir = base_dir + "train/" 
work_dir = root_dir + "dataset/" 
if os.path.exists(work_dir): 
 remove_tree(work_dir) 
 
os.mkdir(work_dir) 
copy_tree(train_dir, work_dir) 
copy_tree(test_dir, work_dir) 
print("Working Directory Contents:", os.listdir(work_dir)) 
WORK_DIR = './dataset/' 
CLASSES = [ 'NonDemented', 
 'VeryMildDemented', 
 'MildDemented', 
 'ModerateDemented'] 
IMG_SIZE = 176 
IMAGE_SIZE = [176, 176] 
DIM = (IMG_SIZE, IMG_SIZE) 
#Performing Image Augmentation to have more data samples 
ZOOM = [.99, 1.01] 
BRIGHT_RANGE = [0.8, 1.2] 
HORZ_FLIP = True 
FILL_MODE = "constant" 
DATA_FORMAT = "channels_last" 
work_dr = IDG(rescale = 1./255, 
brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, 
data_format=DATA_FORMAT, fill_mode=FILL_MODE, 
horizontal_flip=HORZ_FLIP) 
train_data_gen = 
work_dr.flow_from_directory(directory=WORK_DIR, 
target_size=DIM, batch_size=6500, shuffle=False) 
def show_images(generator,y_pred=None): 
 """
nput: An image generator,predicted labels (optional) 
 Output: Displays a grid of 9 images with lables 
 """ 
 
 # get image lables 
 labels =dict(zip([0,1,2,3], CLASSES)) 
 
 # get a batch of images 
 x,y = generator.next() 
 
 # display a grid of 9 images 
 plt.figure(figsize=(10, 10)) 
 if y_pred is None: 
 for i in range(9): 
 ax = plt.subplot(3, 3, i + 1) 
 idx = randint(0, 6400) 
 plt.imshow(x[idx]) 
 plt.axis("off") 
 plt.title("Class:{}".format(labels[np.argmax(y[idx])])) 
 
 else: 
 for i in range(9): 
 ax = plt.subplot(3, 3, i + 1) 
 plt.imshow(x[i]) 
 plt.axis("off") 
 plt.title("Actual:{} 
\nPredicted:{}".format(labels[np.argmax(y[i])],labels[y_pred[i]])) 
 # Display Train Images 
show_images(train_data_gen) 
train_data, train_labels = train_data_gen.next() 
print(train_data.shape, train_labels.shape) 
#Performing over-sampling of the data, since the classes are 
imbalanced 
sm = SMOTE(random_state=42) 
train_data, train_labels = sm.fit_resample(train_data.reshape(-1, 
IMG_SIZE * IMG_SIZE * 3), train_labels)
train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3) 
print(train_data.shape, train_labels.shape) 
Splitting the data into train, test, and validation sets 
train_data, test_data, train_labels, test_labels = 
train_test_split(train_data, train_labels, test_size = 0.2, 
random_state=42) 
train_data, val_data, train_labels, val_labels = 
train_test_split(train_data, train_labels, test_size = 0.2, 
random_state=42) 
def conv_block(filters, act='relu'): 
 """Defining a Convolutional NN block for a Sequential CNN 
model. """ 
 
 block = Sequential() 
 block.add(Conv2D(filters, 3, activation=act, padding='same')) 
 block.add(Conv2D(filters, 3, activation=act, padding='same')) 
 block.add(BatchNormalization()) 
 block.add(MaxPool2D()) 
 
 return block 
def dense_block(units, dropout_rate, act='relu'): 
 """Defining a Dense NN block for a Sequential CNN model. """ 
 
 block = Sequential() 
 block.add(Dense(units, activation=act)) 
 block.add(BatchNormalization()) 
 block.add(Dropout(dropout_rate)) 
 
 return block 
def construct_model(act='relu'): 
 """Constructing a Sequential CNN architecture for performing 
the classification task. """ 
 
 model = Sequential([ 
 Input(shape=(*IMAGE_SIZE, 3)), 
 Conv2D(16, 3, activation=act, padding='same'), 
 Conv2D(16, 3, activation=act, padding='same'), 
 MaxPool2D(), 
 conv_block(32),
conv_block(64), 
 conv_block(128), 
 Dropout(0.2), 
 conv_block(256), 
 Dropout(0.2), 
 Flatten(), 
 dense_block(512, 0.7), 
 dense_block(128, 0.5), 
 dense_block(64, 0.3), 
 Dense(4, activation='softmax') 
 ], name = "cnn_model") 
return model 
class MyCallback(tf.keras.callbacks.Callback): 
 def on_epoch_end(self, epoch, logs={}): 
 if logs.get('val_acc') > 0.99: 
 print("\nReached accuracy threshold! Terminating training.") 
 self.model.stop_training = True 
 
my_callback = MyCallback() 
#EarlyStopping callback to make sure model is always learning 
early_stopping = EarlyStopping(monitor='val_loss', patience=2) 
#Defining other parameters for our CNN model 
model = construct_model() 
METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'), 
 tf.keras.metrics.AUC(name='auc'), 
 tfa.metrics.F1Score(num_classes=4)] 
CALLBACKS = [my_callback] 
 
model.compile(optimizer='adam', 
 loss=tf.losses.CategoricalCrossentropy(), 
 metrics=METRICS) 
model.summary() 
EPOCHS = 100
history = model.fit(train_data, train_labels, 
validation_data=(val_data, val_labels), callbacks=CALLBACKS, 
epochs=EPOCHS) 
#Plotting the trend of the metrics during training 
fig, ax = plt.subplots(1, 3, figsize = (30, 5)) 
ax = ax.ravel() 
for i, metric in enumerate(["acc", "auc", "loss"]): 
 ax[i].plot(history.history[metric]) 
 ax[i].plot(history.history["val_" + metric]) 
 ax[i].set_title("Model {}".format(metric)) 
 ax[i].set_xlabel("Epochs") 
 ax[i].set_ylabel(metric) 
 ax[i].legend(["train", "val"]) 
#Evaluating the model on the data 
#train_scores = model.evaluate(train_data, train_labels) 
#val_scores = model.evaluate(val_data, val_labels) 
test_scores = model.evaluate(test_data, test_labels) 
#print("Training Accuracy: %.2f%%"%(train_scores[1] * 100)) 
#print("Validation Accuracy: %.2f%%"%(val_scores[1] * 100)) 
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100)) 
#Predicting the test data 
pred_labels = model.predict(test_data) 
#Print the classification report of the tested data 
#Since the labels are softmax arrays, we need to roundoff to have it 
in the form of 0s and 1s, 
#similar to the test_labels 
def roundoff(arr): 
 """To round off according to the argmax of each predicted label 
array. """ 
 arr[np.argwhere(arr != arr.max())] = 0 
 arr[np.argwhere(arr == arr.max())] = 1 
 return arr 
for labels in pred_labels:
labels = roundoff(labels) 
print(classification_report(test_labels, pred_labels, 
target_names=CLASSES)) 
Plot the confusion matrix to understand the classification in detail 
pred_ls = np.argmax(pred_labels, axis=1) 
test_ls = np.argmax(test_labels, axis=1) 
conf_arr = confusion_matrix(test_ls, pred_ls) 
plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
ax = sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', 
xticklabels=CLASSES, yticklabels=CLASSES) 
plt.title('Alzheimer\'s Disease Diagnosis') 
plt.xlabel('Prediction') 
plt.ylabel('Truth') 
plt.show(ax) 
#Printing some other classification metrics 
print("Balanced Accuracy Score: {} %".format(round(BAS(test_ls, 
pred_ls) * 100, 2))) 
print("Matthew's Correlation Coefficient: {} 
%".format(round(MCC(test_ls, pred_ls) * 100, 2))) 
#Saving the model for future use 
model_dir = work_dir + "alzheimer_cnn_model" 
model.save(model_dir, save_format='h5') 
os.listdir(work_dir) 
pretrained_model = tf.keras.models.load_model(model_dir) 
#Check its architecture 
plot_model(pretrained_model, to_file=work_dir + 
"model_plot.png", show_shapes=True, show_layer_names=True) 
#Flask App 
import os 
import urllib.request
from flask import Flask, flash, request, redirect, url_for, 
render_template 
from werkzeug.utils import secure_filename 
import numpy as np 
import tensorflow as tf 
ASSET_FOLDER = 'static/assets/' 
import os 
import urllib.request 
from flask import Flask, flash, request, redirect, url_for, 
render_template 
from werkzeug.utils import secure_filename 
import numpy as np 
import tensorflow as tf 
app = Flask(_name_) 
app.config['ASSET_FOLDER'] = ASSET_FOLDER 
github_icon = os.path.join(app.config['ASSET_FOLDER'], 
'github.png') 
linkedin_icon = os.path.join(app.config['ASSET_FOLDER'], 
'linkedin.png') 
instagram_icon = os.path.join(app.config['ASSET_FOLDER'], 
'instagram.png') 
twitter_icon = os.path.join(app.config['ASSET_FOLDER'], 
'twitter.png') 
bg_video = os.path.join(app.config['ASSET_FOLDER'], 'pexelstima-miroshnichenko-6010766.mp4') 
 
@app.route('/') 
def upload_form(): 
 return render_template('index.html', type="", 
githubicon=github_icon, linkedinicon=linkedin_icon, 
instagramicon=instagram_icon, twittericon=twitter_icon, 
bgvideo=bg_video) 
@app.route('/', methods=['POST']) 
def upload_image(): 
 file = request.files['file'] 
 print(file.filename) 
 file.save(os.path.join("static/uploads", 
secure_filename(file.filename)))
imvar = 
tf.keras.preprocessing.image.load_img(os.path.join("static/uploads", 
secure_filename(file.filename))).resize((176, 176)) 
 imarr = tf.keras.preprocessing.image.img_to_array(imvar) 
 imarr = np.array([imarr]) 
 model2 = tf.keras.models.load_model("model") 
 impred = model2.predict(imarr) 
 def roundoff(arr): 
 """To round off according to the argmax of each predicted 
label array. """ 
 arr[np.argwhere(arr != arr.max())] = 0 
 arr[np.argwhere(arr == arr.max())] = 1 
 return arr 
 for classpreds in impred: 
 impred = roundoff(classpreds) 
 
 classcount = 1 
 for count in range(4): 
 if impred[count] == 1.0: 
 break 
 else: 
 classcount+=1 
 
 classdict = {1: "Mild Dementia", 2: "Moderate Dementia", 3: "No 
Dementia, Patient is Safe", 4: "Very Mild Dementia"} 
 print([classcount]) 
 c = 'xyz' 
 return render_template('index.html', type="Patient is suffering 
from "+str(classdict[classcount]), githubicon=github_icon, 
linkedinicon=linkedin_icon, instagramicon=instagram_icon, 
twittericon=twitter_icon, bgvideo=bg_video) 
if _name_ == "_main_": 
 app.run()
