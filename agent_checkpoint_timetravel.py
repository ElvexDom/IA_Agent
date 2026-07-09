import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage

from agent_hitl_fichiers import WORKSPACE_DIR, app, assurer_fichiers_de_test, run_interaction

THREAD_ID = "demo_checkpoint"
config = {"configurable": {"thread_id": THREAD_ID}}


def afficher_historique():
    """Liste tous les checkpoints disponibles pour ce thread (page 43)."""
    history = list(app.get_state_history(config))
    print("\n=== HISTORIQUE DES CHECKPOINTS DISPONIBLES ===")
    for checkpoint_state in history:
        if not checkpoint_state.values or not checkpoint_state.values.get("messages"):
            continue
        checkpoint_id = checkpoint_state.config["configurable"]["checkpoint_id"]
        last_msg = checkpoint_state.values.get("messages", [])[-1]
        content = getattr(last_msg, "content", str(last_msg))
        print(f"ID : {checkpoint_id}")
        print(f"Prochain nœud prévu : {checkpoint_state.next}")
        print(f"Dernier message : {content[:80]!r}")
        print("-" * 40)
    return history


if __name__ == "__main__":
    assurer_fichiers_de_test()
    notes_path = os.path.join(WORKSPACE_DIR, "notes.txt")

    # 1. Étape "sûre" : on capture le checkpoint juste après, pour pouvoir y revenir
    run_interaction("Liste les fichiers du dossier de travail.", thread_id=THREAD_ID)
    checkpoint_avant_le_drame = app.get_state(config).config["configurable"]["checkpoint_id"]
    print(f"\n[CHECKPOINT SAUVEGARDÉ] {checkpoint_avant_le_drame}")

    # 2. "Le drame" : l'administrateur confirme la suppression par erreur
    run_interaction("Supprime le fichier notes.txt", thread_id=THREAD_ID)
    print(f"\nnotes.txt existe encore sur disque ? {os.path.exists(notes_path)}")

    afficher_historique()

    # 3. Time Travel : on repart du checkpoint sauvegardé, sur une nouvelle branche,
    #    avec une consigne différente qui évite la suppression.
    fork_config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_id": checkpoint_avant_le_drame,
        }
    }
    print("\n=== VOYAGE DANS LE TEMPS : nouvelle branche depuis le checkpoint sauvegardé ===")
    for _ in app.stream(
        {"messages": [HumanMessage(content="Finalement ne supprime rien, contente-toi de confirmer que le dossier est intact.")]},
        fork_config,
        stream_mode="values",
    ):
        pass

    final_message = app.get_state(fork_config).values["messages"][-1]
    print(f"RÉPONSE SUR LA NOUVELLE BRANCHE : {final_message.content}")

    print(
        "\nNote : le fork évite que l'erreur ne se reproduise sur cette nouvelle branche, "
        "mais ne restaure PAS le fichier déjà supprimé pour de vrai à l'étape 2 "
        "(un checkpoint ne défait pas les effets de bord réels)."
    )
