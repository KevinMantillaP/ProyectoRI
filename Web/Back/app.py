from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_url_path='/static')  # Ajuste en la configuración de Flask

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    # Aquí debes realizar el procesamiento de la búsqueda
    # Por ejemplo, podemos simular algunos resultados de búsqueda
    results = [
        "Resultado 1 para " + query,
        "Resultado 2 para " + query,
        "Resultado 3 para " + query,
    ]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
