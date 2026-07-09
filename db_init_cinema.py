import sqlite3


def init_db():
    conn = sqlite3.connect("cinema.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        director TEXT NOT NULL,
        year INTEGER NOT NULL
    );
    """)

    # Nettoyage pour éviter les doublons au ré-allumage
    cursor.execute("DELETE FROM movies;")

    mock_data = [
        ("Inception", "Christopher Nolan", 2010),
        ("Interstellar", "Christopher Nolan", 2014),
        ("Oppenheimer", "Christopher Nolan", 2023),
        ("Pulp Fiction", "Quentin Tarantino", 1994),
        ("Kill Bill", "Quentin Tarantino", 2003),
        ("Taxi Driver", "Martin Scorsese", 1976),
        ("Goodfellas", "Martin Scorsese", 1990),
    ]
    cursor.executemany(
        "INSERT INTO movies (title, director, year) VALUES (?, ?, ?);", mock_data
    )

    conn.commit()
    conn.close()
    print("Base de donnees 'cinema.db' initialisee avec succes !")


if __name__ == "__main__":
    init_db()
