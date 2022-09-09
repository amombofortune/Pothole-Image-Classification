"""
from crypt import methods
from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn


app = Flask(__name__)

#SimpleNet model
class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

simplenet = SimpleNet()
simplenet_state_dict = torch.load("/Users/fortuneamombo/Desktop/Flasktutorial/model.pth")
simplenet.load_state_dict(simplenet_state_dict)

img_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

#Copy model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

simplenet.to(device)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
   imagefile = request.files['imagefile'] 
   image_path = "./images/" + imagefile.filename
   imagefile.save(image_path)

    #Load our image and preprocess it to the correct shape that the model wants. Different ways to load image
   image = Image.open(image_path)
    #Convert image to array
   image = img_transforms(image).to(device)
   image = torch.unsqueeze(image, 0)

   #Making predictions
   labels = ['normal','pothole']
    
   simplenet.eval()
   prediction = F.softmax(simplenet(image), dim=1)
   prediction = prediction.argmax()
   print(labels[prediction])

   return render_template('index.html', prediction1=labels[prediction])
   #return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)


"""
"""
from crypt import methods
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.models import load_model


app = Flask(__name__)

model = load_model('model.h5')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
   imagefile = request.files['imagefile'] 
   image_path = "./images/" + imagefile.filename
   imagefile.save(image_path)

    #Load our image and preprocess it to the correct shape that the model wants. Different ways to load image
   image = tf.keras.utils.load_img(image_path, target_size=(128, 128))
    #Convert image to array
   image = tf.keras.utils.img_to_array(image)
    #Reshape image
   image = np.expand_dims(image, axis = 0)
    #Predict
   result = model.predict(image)
    #Label
   if result[0][0] == 1:
    prediction = 'pothole'
   else:
    prediction = 'normal'

   print(prediction)
   return render_template('index.html', prediction1=prediction)
   #return render_template('index.html')

  

if __name__ == "__main__":
    app.run(debug=True)

"""
