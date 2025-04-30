import sqlite3
import csv
from datetime import datetime

import click
from flask import current_app, g


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)


def get_db():
    print(current_app.config['DATABASE'])
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

    categories_data = {
        1: 'U Unterrichtszeit',
        2: 'A Vertretungsstunden',
        3: 'A Aufsichten',
        4: 'U Korrekturzeiten',
        5: 'U Unterrichtsvor- und Nachbereitung',
        6: 'U Abschlussprüfungen',
        7: 'F Organisatorische Tätigkeiten als Klassenlehrkraft, Tutor*in, Beratungslehrkraft',
        8: 'F Funktionsarbeit (mit und ohne F-Zeiten)',
        9: 'F Schulleitungsfunktionen (mit und ohne F-Zeiten)',
        10: 'F Schulleitungsfunktionen - Leitung und Personalführung',
        11: 'F Schulleitungsfunktionen - Administration, Qualitätssicherung und Schulorganisation',
        12: 'F Schulleitungsfunktionen - Pädagogische Gestaltung und Entwicklung',
        13: 'F Schulleitungsfunktionen - Pädagogische Intervention/ Krisenintervention',
        14: 'A Konferenzen / Sitzungen',
        15: 'F Arbeitsgruppe / Ausschuss',
        16: 'U Pädagogische Kommunikation I – Eltern- und Schülergespräche',
        17: 'U Pädagogische Kommunikation II – Klassen und Zeugniskonferenz',
        18: 'U Pädagogische Kommunikation III – Als Fachlehrer*in mit Kolleg*innen, anderen Professionen und Externen',
        19: 'F Pädagogische Kommunikation IV – Im Rahmen einer Funktion: Klassenlehrkraft, Tutor*in, Beratungslehrkraft',
        20: 'Sonstige Kommunikation (neu)',
        21: 'A Außerunterrichtliche schulische Veranstaltungen (Fahrten / Veranstaltungen) ohne Übernachtung',
        22: 'Außerunterrichtliche schulische Veranstaltungen (Fahrten / Veranstaltungen) mit Übernachtung (neu)',
        23: 'U Arbeitsplatzorganisation',
        24: 'U Arbeitsraum Schulgebäude (unterrichtsbezogen)',
        25: 'A Weiterbildungszeiten (Fortbildungen)',
        26: 'Krankheitstag (neu)',
        27: 'Arztbesuch (neu)',
        28: 'Sonderurlaub (neu)',
        29: 'Sonstiges (neu)',
        30: 'Zeiterfassungszeit (neu)',
        31: 'Wege zwischen Schulen I',
        32: 'Wege zwischen Schulen II',
        33: 'Wege zwischen Schulen III',
    }

    for category_id, category_name in categories_data.items():
        db.execute("INSERT INTO categories (id, name) VALUES (?, ?)", (category_id, category_name))

    # Import data from categorize.csv
    csv_file = current_app.open_resource('categorize.csv', mode='rt')
    reader = csv.DictReader(csv_file, delimiter=';')
    for row in reader:
        db.execute(
            "INSERT INTO all_data (id, further_remarks) VALUES (?, ?)",
            (row['id'], row['further_remarks'])
        )

    db.commit()


@click.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


sqlite3.register_converter(
    "timestamp", lambda v: datetime.fromisoformat(v.decode())
)
