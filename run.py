import flask
from flask import request, jsonify

from app.handler import Handler

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods = ['GET'])
def getHome():
    handler = Handler()
    choices = {
        'feature': 'ngrams',
        'classifier': 'nb'
    }
    try:
        res = handler.classifyRequest(choices)
        return res, 201
    except:
        res = {
            'message' : 'Some error'
        }
        return res, 500
    #return handler.cleanRequest('hehe')


app.run(host = '0.0.0.0', port = '8080')