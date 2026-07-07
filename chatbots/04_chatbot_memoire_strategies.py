from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOllama(model="llama3.2:3b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant technique pro. Réponds de manière concise."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

chain = prompt | llm

# ============================================================
# UTILITAIRE : Comptage de tokens (approximation sans transformers)
# ============================================================
def count_tokens(history: list) -> int:
    """Approximation : 1 token ≈ 0.75 mot."""
    texte = " ".join([m.content for m in history])
    return int(len(texte.split()) / 0.75)


# ============================================================
# UTILITAIRE : Boucle commune
# ============================================================
def get_input() -> str:
    return input("\nClient : ")

def is_stop(text: str) -> bool:
    return text.lower().strip() in ["exit", "stop", "fin"]

def ask(user_input: str, history: list) -> str:
    response = chain.invoke({
        "input": user_input,
        "chat_history": history
    })
    return response.content


# ============================================================
# STRATÉGIE 1 — FULL BUFFER (Tout garder)
# ============================================================
def chatbot_full_buffer():
    """Envoie tout l'historique à chaque échange. Précision max, coût max."""
    print("🤖 Full Buffer démarré. (exit/stop/fin pour quitter)\n")

    history = []

    while True:
        user_input = get_input()
        if is_stop(user_input):
            print("🤖 Bot : À bientôt !")
            break

        bot_reply = ask(user_input, history)

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=bot_reply))

        print(f"🤖 Bot : {bot_reply}")
        print(f"   📊 Tokens totaux : {count_tokens(history)} | Messages : {len(history)}")


# ============================================================
# STRATÉGIE 2 — SLIDING WINDOW (Fenêtre glissante)
# ============================================================
def chatbot_sliding_window(k: int = 6):
    """Garde uniquement les k derniers messages."""
    print(f"🤖 Sliding Window (k={k}) démarré. (exit/stop/fin pour quitter)\n")

    history = []

    while True:
        user_input = get_input()
        if is_stop(user_input):
            print("🤖 Bot : À bientôt !")
            break

        bot_reply = ask(user_input, history)

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=bot_reply))

        # 🔑 On ne garde que les k derniers messages
        history = history[-k:]

        print(f"🤖 Bot : {bot_reply}")
        print(f"   📊 Tokens estimés : {count_tokens(history)} | Messages en mémoire : {len(history)}")


# ============================================================
# STRATÉGIE 3 — SUMMARY BUFFER (Résumé glissant)
# ============================================================
def chatbot_summary_buffer():
    """Résume tout l'historique à chaque échange."""
    print("🤖 Summary Buffer démarré. (exit/stop/fin pour quitter)\n")

    summary = ""
    history = []

    while True:
        user_input = get_input()
        if is_stop(user_input):
            print("🤖 Bot : À bientôt !")
            break

        if summary:
            history = [AIMessage(content=f"[Résumé de la conversation]: {summary}")]

        bot_reply = ask(user_input, history)

        message_sortant = f"User: {user_input} | Assistant: {bot_reply}"
        summary = llm.invoke(
            f"Update ce résumé : '{summary}' avec cette nouvelle info : '{message_sortant}'. "
            f"Sois très concis, garde uniquement les faits importants."
        ).content

        print(f"🤖 Bot : {bot_reply}")
        print(f"   📊 Tokens résumé : {count_tokens([AIMessage(content=summary)])}")
        print(f"   📝 Résumé actuel : {summary[:100]}...")


# ============================================================
# STRATÉGIE 4 — MOVING SUMMARY (Fenêtre + Résumé)
# ============================================================
def chatbot_moving_summary(k: int = 4):
    """Garde les k derniers messages en clair + résumé du passé."""
    print(f"🤖 Moving Summary (k={k}) démarré. (exit/stop/fin pour quitter)\n")

    history = []
    summary = ""

    while True:
        user_input = get_input()
        if is_stop(user_input):
            print("🤖 Bot : À bientôt !")
            break

        context = []
        if summary:
            context.append(AIMessage(content=f"[Résumé du passé]: {summary}"))
        context.extend(history)

        bot_reply = ask(user_input, context)

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=bot_reply))

        if len(history) > k:
            messages_a_resumer = history[:-k]
            history = history[-k:]

            texte = " ".join([m.content for m in messages_a_resumer])
            summary = llm.invoke(
                f"Update ce résumé : '{summary}' avec ces échanges : '{texte}'. "
                f"Garde uniquement les faits importants, sois concis."
            ).content

        print(f"🤖 Bot : {bot_reply}")
        print(f"   📊 Tokens fenêtre : {count_tokens(history)} | Messages récents : {len(history)}")
        if summary:
            print(f"   📝 Résumé passé  : {summary[:80]}...")


# ============================================================
# STRATÉGIE 5 — COMPACTAGE INTELLIGENT (style Claude)
# ============================================================
def chatbot_compactage(seuil_tokens: int = 500):
    """
    Compacte l'historique uniquement quand on approche du seuil.
    Préserve : faits importants, code, décisions, tâches en cours.
    """
    print(f"🤖 Compactage Intelligent (seuil={seuil_tokens} tokens) démarré. (exit/stop/fin pour quitter)\n")

    history = []

    PROMPT_COMPACTAGE = """Tu es un expert en compression de contexte.
Analyse cet historique et produis un résumé structuré en préservant OBLIGATOIREMENT :
- 📌 Les faits importants (noms, dates, chiffres)
- 💻 Tout le code produit ou mentionné
- ✅ Les décisions prises
- 🎯 Les tâches en cours ou demandées
- ❓ Les questions sans réponse

Historique :
{historique}

Résumé structuré :"""

    def compacter(history: list) -> list:
        texte = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history
        ])
        resume = llm.invoke(PROMPT_COMPACTAGE.format(historique=texte)).content

        print(f"\n   🗜️  COMPACTAGE DÉCLENCHÉ")
        print(f"   📉 {len(history)} messages → 1 résumé structuré")
        print(f"   📝 Aperçu : {resume[:100]}...\n")

        return [AIMessage(content=f"[CONTEXTE COMPACTÉ]\n{resume}")]

    while True:
        user_input = get_input()
        if is_stop(user_input):
            print("🤖 Bot : À bientôt !")
            break

        bot_reply = ask(user_input, history)

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=bot_reply))

        tokens = count_tokens(history)
        if tokens >= seuil_tokens:
            history = compacter(history)

        print(f"🤖 Bot : {bot_reply}")
        print(f"   📊 Tokens : {tokens} / {seuil_tokens} | Messages : {len(history)}")


# ============================================================
# MENU PRINCIPAL
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("       🤖 CHATBOT — STRATÉGIES MÉMOIRE")
    print("=" * 50)
    print("  1 - Full Buffer     (tout garder)")
    print("  2 - Sliding Window  (k derniers messages)")
    print("  3 - Summary Buffer  (résumé à chaque tour)")
    print("  4 - Moving Summary  (fenêtre + résumé)")
    print("  5 - Compactage      (résumé structuré sur seuil)")
    print("=" * 50)

    choix = input("\nVotre choix : ").strip()

    strategies = {
        "1": chatbot_full_buffer,
        "2": lambda: chatbot_sliding_window(k=6),
        "3": chatbot_summary_buffer,
        "4": lambda: chatbot_moving_summary(k=4),
        "5": lambda: chatbot_compactage(seuil_tokens=500),
    }

    if choix in strategies:
        strategies[choix]()
    else:
        print("❌ Choix invalide.")