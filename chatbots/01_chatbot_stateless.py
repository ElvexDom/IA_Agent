import ollama

def chatbot():
    print("🤖 Chatbot démarré. (Tape 'exit', 'stop' ou 'fin' pour quitter)\n")
    
    while True:
        user_input = input("\nClient : ")
        
        # Condition d'arrêt
        if user_input.lower().strip() in ["exit", "stop", "fin"]:
            print("🤖 Bot : À bientôt !")
            break
        
        # Appel à Ollama
        response = ollama.generate(
            model="llama3.2:3b",
            prompt=user_input
        )
        
        print(f"🤖 Bot : {response['response']}")

if __name__ == "__main__":
    chatbot()