import sqlite3


def init_db():
    conn = sqlite3.connect("boutique.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        order_date TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """)

    # Nettoyage pour éviter les doublons au ré-allumage
    cursor.execute("DELETE FROM orders;")
    cursor.execute("DELETE FROM users;")

    users = [(1, "Alice Dupont"), (2, "Bob Martin")]
    cursor.executemany("INSERT INTO users (id, name) VALUES (?, ?);", users)

    # Dates calées par rapport à "aujourd'hui" pour tester les 3 cas de figure
    # (dans les délais / hors délais / limite)
    orders = [
        (101, 1, "Ordinateur portable", "Electronique", "2026-06-20"),      # > 14 jours -> refusé
        (102, 1, "Chaise de bureau", "Mobilier", "2026-06-25"),             # < 30 jours -> accepté
        (103, 2, "Réfrigérateur", "Electromenager", "2026-05-01"),          # > 30 jours -> refusé
        (104, 2, "Casque audio", "Electronique", "2026-07-02"),             # < 14 jours -> accepté
    ]
    cursor.executemany(
        "INSERT INTO orders (id, user_id, product_name, category, order_date) VALUES (?, ?, ?, ?, ?);",
        orders,
    )

    conn.commit()
    conn.close()
    print("Base de donnees 'boutique.db' initialisee avec succes !")


if __name__ == "__main__":
    init_db()
