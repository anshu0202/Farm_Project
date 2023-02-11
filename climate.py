import requests
from index import  predict_value

#This module will fetch climate details from OpenWheather of the given place by the user using api key 


def climate_details(loc):
  api_key = "f05b987431e2f8ac0fd0991f2146aa10"


  # currently we are getting values only for the London,UK
  location = "London,UK"
  weather_data = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}").json()
  humidity=weather_data['main']['humidity']
  temp=weather_data['main']['temp']
  pressure=weather_data['main']['pressure']
  print("************************")
  print("Crop details :")
  crop_name=predict_value()
  print("Crop name : ",crop_name)
  print("Location : ",loc)
  print("Temperature : ",temp)
  print("Humidity : ",humidity)
  print("Pressure : ",pressure)
 


# print("Humidity is ",humidity)
# print("Temp is ",temp)
# print("Pressure is ",pressure)


