from flask import Flask,request,render_template,send_from_directory,send_file
from flask_cors import CORS
from utils import DenseAutoEncoderSearch

app = Flask(__name__)
CORS(app=app)

DAESearch = DenseAutoEncoderSearch()

# Serving Static
@app.route("/static/<string:_type>/<string:_file>",methods=['GET'])
def serve_static(_type,_file):
    print (_type,_file)
    return send_from_directory(f"./templates/static/{_type}",_file,mimetype=f"text/{_type}")

@app.route("/images/<string:name>")
def image_serve(name):
    return send_file(f"./images/{name}")

@app.route("/",methods=['GET'])
def index():
    return "Hello"

@app.route("/search",methods=['GET','POST'])
def search():
    img = request.files['image']
    img.save(f"./search/{img.filename}")
    results = DAESearch(img.filename)
    results = [{"path":f"http://localhost:8080/images/img_{i}.jpg","name":f"img_{i}.jpg"} for i in results ][:50]
    return {
        "results":results
    }

if __name__ == "__main__":
    app.run(port=8080,threaded=True)