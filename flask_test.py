from flask import Flask, request
app = Flask(__name__)

@app.route('/player', methods=['POST'])
def receive_player():
    data = request.json
    print(f"Received: {data['player']} ({data['confidence']}) ({data['timestamp']})")
    # Do whatever you need with the data
    return {'status': 'ok'}

app.run(host='0.0.0.0', port=5000)