
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

# Initialize Flask application
app = Flask(__name__)

# Define the labels for the GoEmotions dataset
labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Specify the model ID and file name for the ONNX model
model_id = "SamLowe/roberta-base-go_emotions-onnx"
file_name = "onnx/model_quantized.onnx"

# Load the ONNX model and tokenizer
model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create a pipeline for text classification
onnx_classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    function_to_apply="sigmoid"
)

@app.route("/")
def home():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>RoBERTa Sentiment Analysis with GoEmotions</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                margin: 20px;
            }
            .container {
                max-width: 800px;
                margin: auto;
                background-color: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h2 {
                color: #333333;
                text-align: center;
            }
            form {
                text-align: center;
            }
            textarea {
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                resize: vertical;
            }
            input[type="submit"] {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                padding: 10px;
                margin-bottom: 5px;
                background-color: #f9f9f9;
                border-left: 6px solid #4CAF50;
                border-radius: 4px;
            }
            li strong {
                color: #333333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>RoBERTa Sentiment Analysis with GoEmotions</h2>
            <form action="/predict" method="post">
                <label for="text">Enter text:</label><br><br>
                <textarea id="text" name="text" rows="4" cols="50"></textarea><br><br>
                <input type="submit" value="Submit">
            </form>
        </div>
    </body>
    </html>
    '''
    return html

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentences = [text]
    model_outputs = onnx_classifier(sentences)
    
    results = []
    for output in model_outputs:
        for res in output:
            results.append({"label": res['label'], "score": res['score']})

    # Prepare HTML response
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>RoBERTa Sentiment Analysis with GoEmotions</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                margin: 20px;
            }
            .container {
                max-width: 800px;
                margin: auto;
                background-color: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h2 {
                color: #333333;
                text-align: center;
            }
            form {
                text-align: center;
            }
            textarea {
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                resize: vertical;
            }
            input[type="submit"] {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                padding: 10px;
                margin-bottom: 5px;
                background-color: #f9f9f9;
                border-left: 6px solid #4CAF50;
                border-radius: 4px;
            }
            li strong {
                color: #333333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>RoBERTa Sentiment Analysis with GoEmotions</h2>
            <form action="/predict" method="post">
                <label for="text">Enter text:</label><br><br>
                <textarea id="text" name="text" rows="4" cols="50">{{ text }}</textarea><br><br>
                <input type="submit" value="Submit">
            </form>
            <div>
                <h3>Prediction Results for: <em>{{ text }}</em></h3>
                <ul>
                    {% for result in results %}
                        <li><strong>{{ result.label }}:</strong> {{ result.score }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html, text=text, results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
