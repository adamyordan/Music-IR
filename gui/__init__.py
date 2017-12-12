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
    return jsonify(vsm.search_api(q))

def start_gui():
    app.run(debug=True)
