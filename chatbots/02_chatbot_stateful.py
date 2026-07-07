import ollama

def chatbot_memory():
    print("🤖 Chatbot avec mémoire démarré. (Tape 'exit', 'stop' ou 'fin' pour quitter)\n")
    
    # Historique initialisé avec le message système
    messages = [
        {
            'role': 'system',
            'content': 'Tu es un assistant technique professionnel. Tu dois aider le client en te basant sur les informations fournies dans la discussion.'
        }
    ]
    
    while True:
        user_input = input("\nClient : ")
        
        # Condition d'arrêt
        if user_input.lower().strip() in ["exit", "stop", "fin"]:
            print("🤖 Bot : À bientôt !")
            break
        
        # Ajout du message utilisateur dans l'historique
        messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Appel à Ollama avec tout l'historique
        response = ollama.chat(
            model="llama3.2:3b",
            messages=messages
        )
        
        # Récupération de la réponse
        bot_reply = response['message']['content']
        
        # Ajout de la réponse du bot dans l'historique
        messages.append({
            'role': 'assistant',
            'content': bot_reply
        })
        
        print(f"🤖 Bot : {bot_reply}")

if __name__ == "__main__":
    chatbot_memory()