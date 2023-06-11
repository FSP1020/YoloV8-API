from flask import Flask, jsonify

# Create the Flask application
app = Flask(__name__)

# Define a basic route
@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/api/TEST', methods=['GET'])
def TEST_API():

    return jsonify("TEST")

# Run the application
if __name__ == '__main__':
    app.run()
