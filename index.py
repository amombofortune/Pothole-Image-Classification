import PIL.Image
from gmplot import gmplot
import PIL.ExifTags
from geopy.geocoders import Nominatim
import webbrowser


img = PIL.Image.open("./images/test5.jpeg")




#Get the metadata of the image
exif = {
    PIL.ExifTags.TAGS[k]: v
    for k, v in img._getexif().items()
    if k in PIL.ExifTags.TAGS
}

#print(exif['GPSInfo'])
#Only pick the location coordinates 
north = exif['GPSInfo'] [2]
west = exif['GPSInfo'] [4]

#print(north)
#print(west)

#convert these values into latitude and longitude
lat = ((((north[0] * 60) + north[1]) * 60) + north[2]) / 60 / 60
long = ((((west[0] * 60) + west[1]) * 60) + west[2]) / 60 / 60

#we get a fraction. We want a decimal point so we convert to float
lat, long = float(lat), float(long)

#print('Latitude:',lat)
#print('Longitude',long)


#Draws an estimate of the location
gmap = gmplot.GoogleMapPlotter(lat, long, 12)
gmap.marker(lat, long, "cornflowerblue")
gmap.draw("location.html")


#Get specific location
geoLoc = Nominatim(user_agent="GetLoc")
locname = geoLoc.reverse(f"{lat}, {long}")

print(locname.address)

webbrowser.open("location.html", new=2)



