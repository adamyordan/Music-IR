from flask import Flask, render_template, jsonify, request
from ir.se import get_vsm

app = Flask('music-ir-gui', template_folder='gui/templates')
vsm = get_vsm('ir/data')

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/api/search")
def api_search():
    q = request.args.get('q')
    titleWeight = request.args.get('title')
    artistWeight = request.args.get('artist')
    genreWeight = request.args.get('weight')
    return jsonify(vsm.search(q, titleWeight, artistWeight, genreWeight))

def start_gui():
    app.run(debug=False)
