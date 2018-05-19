from flask import Flask, render_template
from flask_sockets import Sockets

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

@sockets.route('/echo')
def echo_socket(ws):
    while not ws.closed:
        message = ws.receive()
        ws.send(message)

if __name__ == "__main__":
    app.debug = True
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
