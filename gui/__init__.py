from flask import Flask, render_template

app = Flask('music-ir-gui', template_folder='gui/templates')

@app.route("/")
def hello():
    return render_template('index.html')

def start_gui():
    app.run(debug=True)
