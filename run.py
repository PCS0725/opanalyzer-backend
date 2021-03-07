import flask
from flask import request, jsonify

from app.handler import Handler

app = flask.Flask(__name__)
app.config["DEBUG"] = True

handler = Handler()


@app.route('/', methods = ['GET'])
def getHome():
    return '<h1>Hehe</h1>'


@app.route('/classify', methods = ['POST'])
def classifyData():
    choices = request.get_json(force = True)
    # choices = {
    #     'feature': 'ngrams',
    #     'classifier': 'rb'
    # }
    # try:
    #     res = handler.classifyRequest(choices)
    #     return res, 201
    # except:
    #     res = {
    #         'message' : 'Some error'
    #     }
    #     return res, 500
    return handler.classifyRequest(choices)
    #return handler.cleanRequest('hehe')


#work on this: 1mar 2021.
@app.route('/data', methods = ['POST'])
def postData():
    choices = request.get_json(force = True)
    msg = handler.postDataRequest(choices)
    # response = {
    #     'message': msg
    # }
    
    return msg, 201



app.run(host = '0.0.0.0', port = '8080')