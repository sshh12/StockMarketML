from flask import Flask, render_template
from flask_sockets import Sockets

import random
import json

import info

app = Flask(__name__, static_url_path='')
sockets = Sockets(app)

## Load Data ##

main_data = info.main()

###############

@app.route('/')
def index():
    return render_template('index.html', **main_data)

@app.route('/settings')
def settings():
    return render_template('settings.html')

def send(ws, data):
    ws.send(json.dumps(data))

@sockets.route('/prediction')
def prediction_socket(ws):
    while not ws.closed:

        stock = ws.receive().upper()

        if stock in main_data["symbols"]:

            send(ws, {
                'status': 'loading',
                'stock': stock
            })

            # todo prediction here

            r = random.random()
            prediction = [r, 1 - r]
            num_headlines = random.randint(10, 40)

            send(ws, {
                'status': 'complete',
                'stock': stock,
                'prediction': prediction,
                'numheadlines': num_headlines
            })

        else:
            send(ws, {'status': 'error'})



if __name__ == "__main__":
    app.debug = True
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
