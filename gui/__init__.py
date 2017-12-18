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
    lyricsWeight = 0.0 if not request.args.get('lyrics') else request.args.get('lyrics')
    titleWeight = 0.0 if not request.args.get('title') else request.args.get('title')
    artistWeight = 0.0 if not request.args.get('artist') else request.args.get('artist')
    genreWeight = 0.0 if not request.args.get('genre') else request.args.get('genre')
    print lyricsWeight, titleWeight, artistWeight, genreWeight
    return jsonify(vsm.search(q, { 'lyrics': float(lyricsWeight), 'title': float(titleWeight), 'artist': float(artistWeight), 'genre': float(genreWeight) }))

def start_gui():
    app.run(debug=False)
