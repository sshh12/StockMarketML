from flask import Flask, render_template
from flask_sockets import Sockets

import random
import json
import algo

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

            print('Loading ' + stock + '...')

            send(ws, {
                'status': 'loading',
                'stock': stock
            })

            pred, data, time_taken = algo.algo_predict(stock)

            pred_up = round(float(pred[0]), 3)
            pred_down = round(float(pred[1]), 3)

            raw_input = data[-1]
            num_headlines = raw_input.count('**NEXT**')

            send(ws, {
                'status': 'complete',
                'stock': stock,
                'prediction': [pred_up, pred_down],
                'rawinput': raw_input,
                'numheadlines': num_headlines,
                'time': time_taken
            })

        else:
            send(ws, {'status': 'error'})

if __name__ == "__main__":

    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)

    print("http://localhost:5000/")

    server.serve_forever()
