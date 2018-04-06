import os
import time
from datetime import datetime as dt
import re
#from slackclient import SlackClient
from shutil import copy2
import subprocess
import requests

#Make sure to export SLACK_API_TOKEN first
#Then schedule a cron-task

last_date = dt.now()
token = os.environ['SLACK_API_TOKEN']
#sc = SlackClient(token)
chan='#bot-testing'

def daily_post():
    file_name = '{}_tearsheet'.format(dt.now().strftime('%Y-%m-%d'))
    copy2('output.pdf', '{}'.format(file_name))

    message = "Today's Tearsheet"
    attachment = {
        'file': ('{}'.format(file_name), open(file_name, 'rb'), 'pdf')
    }
    payload={
        "filename":file_name, 
        "token":token, 
        "channels":[chan],
    }
    print('request sent')
    r = requests.post("https://slack.com/api/files.upload", params=payload, files=attachment)
    if r.status_code < 400:
        os.remove(file_name)
        print('sucess')
    else:
        return r

daily_post()