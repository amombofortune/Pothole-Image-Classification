
from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import PIL.Image
from gmplot import gmplot
import PIL.ExifTags
from geopy.geocoders import Nominatim
import webbrowser

app = Flask(__name__)

model = load_model('newmodel.h5')

def predict_label(img_path):
    i = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    i = tf.keras.utils.img_to_array(i)
    i = np.expand_dims(i, axis = 0)
    #Predict
    result = model.predict(i)
    #Label
    if result[0][0] == 1:
        prediction = 'Pothole'
    else:
        prediction = 'Plain'

    return prediction



def get_location(img_path):
    im = PIL.Image.open(img_path)
    #Get the metadata of the image
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in im._getexif().items()
        if k in PIL.ExifTags.TAGS
    }

    #Only pick the location coordinates 
    north = exif['GPSInfo'] [2]
    west = exif['GPSInfo'] [4]

    #convert these values into latitude and longitude
    lat = ((((north[0] * 60) + north[1]) * 60) + north[2]) / 60 / 60
    long = ((((west[0] * 60) + west[1]) * 60) + west[2]) / 60 / 60

    #We get a fraction. We want a decimal point so we convert to float
    lat, long = float(lat), float(long)

    #Draws an estimate of the location
    gmap = gmplot.GoogleMapPlotter(lat, long, 12)
    gmap.marker(lat, long, "cornflowerblue")
    gmap.draw("location.html")

    #Get specific location
    geoLoc = Nominatim(user_agent="GetLoc")
    locname = geoLoc.reverse(f"{lat}, {long}")

    webbrowser.open("location.html", new=2)

    return locname.address

    

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route('/submit', methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "./static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

        location = get_location(img_path)

    return render_template("index.html", prediction = p, img_path = img_path, location=location)


if __name__ == '__main__':
    app.run(debug = True)
