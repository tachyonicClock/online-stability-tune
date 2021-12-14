import requests
from datetime import datetime

def notify_me(msg):
    """A utility to notify me when something finishes"""
    url = "https://api.pushover.net/1/messages.json"
    data = {
    "user"  : "uhm4aagjwvaxevybzo329o8abd53i7",
    "token" : "a257znstuy81d5iz5fhg6yatyru9cd",
    "sound" : "magic"
    }
    data["message"] = msg
    data['message'] = data['message'] + "\n" + str(datetime.now())

    r = requests.post(url = url, data = data)
    print(r)
