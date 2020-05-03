from flask import Flask,request,render_template,send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app=app)

# Serving Static

@app.route("/static/<string:_type>/<string:_file>",methods=['GET'])
def serve_static(_type,_file):
    print (_type,_file)
    return send_from_directory(f"./templates/static/{_type}",_file,mimetype=f"text/{_type}")