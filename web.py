from flask import Flask, render_template_string, request, jsonify
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

app = Flask(__name__)

# Load chatbot model and tokenizer
model_name = "facebook/blenderbot-1B-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# HTML template with embedded CSS/JavaScript
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>BlenderBot Chat</title>
    <style>
        body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: Arial; }
        #chatbox { border: 1px solid #ccc; height: 500px; overflow-y: auto; padding: 10px; margin: 20px 0; }
        .user { color: blue; margin: 5px 0; }
        .bot { color: green; margin: 5px 0; }
        input { width: 70%; padding: 10px; }
        button { padding: 10px 20px; background: #4CAF50; color: white; border: none; }
    </style>
</head>
<body>
    <h1>BlenderBot Chat Interface</h1>
    <div id="chatbox"></div>
    <input type="text" id="userInput" placeholder="Type your message...">
    <button onclick="sendMessage()">Send</button>

    <script>
        function addMessage(sender, text) {
            const chatbox = document.getElementById('chatbox');
            const div = document.createElement('div');
            div.className = sender;
            div.innerHTML = `<strong>${sender}:</strong> ${text}`;
            chatbox.appendChild(div);
            chatbox.scrollTop = chatbox.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;

            addMessage('You', message);
            input.value = '';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                const data = await response.json();
                addMessage('Bot', data.reply);
            } catch (error) {
                addMessage('System', 'Error communicating with server');
            }
        }

        // Handle Enter key
        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    inputs = tokenizer([user_input], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    bot_reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return jsonify({'reply': bot_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
