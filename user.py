
## user.py --> climate.py --> index.py 
# The dataset is present in the Crop_recommendation.csv

from climate import climate_details

#This is the starting page of the ML module
print("********************************")
print("This is Home window")
print("This is currently giving the name of crop you should grow at a given place")


print("Enter your location")
location=input()


# This function is define in the climate.py module 
climate_details(location)
