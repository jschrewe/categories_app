from flask import Blueprint, render_template, request, redirect, url_for
import pandas as pd

from .db import get_db
from .random_sample import sample_assigned
from .ml_backend import descriptive_statistics, train_model

# Create the blueprint
bp = Blueprint('machine_learning', __name__, url_prefix='/machine_learning')


def statistics_all():
    db = get_db()

    cursor = db.execute(
        'SELECT * FROM all_data'
    )    

    rows = cursor.fetchall()
    df = pd.DataFrame(
        rows,
        columns=[col[0] for col in cursor.description]
    )
    statistics = descriptive_statistics(df)
    return statistics


@bp.route('/')
def index():
    """
    Display a list of machine learning predictions.
    """
    if not sample_assigned():
        return redirect(url_for('random_sample.index'))

    db = get_db()
    data = db.execute(
        'SELECT * FROM all_data WHERE ml_assigned = 1'
    ).fetchall()

    return render_template('machine_learning/index.html', data=data)


@bp.route('/train')
def train():
    """
    Train the machine learning model and update the database.
    """
    if not sample_assigned():
        return redirect(url_for('random_sample.index'))
    db = get_db()
    # Load the data from the database
    cursor = db.execute(
        'SELECT * FROM all_data WHERE random_sampled = 1 AND assigned = 1'
    )
    rows = cursor.fetchall()

    df = pd.DataFrame(
        rows,
        columns=[col[0] for col in cursor.description]
    )
    statistics = statistics_all()
    # Train the model
    data = train_model(df)

    # Update the database with the trained model
    db.executemany(
        'UPDATE all_data SET ml_assigned = 1, ml_probability = ? WHERE id = ?',
        [(row['ml_probability'], row['id']) for _, row in rows.iterrows()]
    )
    db.commit()

    return render_template(
        'machine_learning/train.html',
        data=rows,
        statistics=statistics
    )


@bp.route('/predict', methods=['POST'])
def predict():
    """
    Simulate machine learning predictions and update the database.
    """
    if not sample_assigned():
        return redirect(url_for('random_sample.index'))

    db = get_db()

    # Example: Simulate predictions by assigning random probabilities
    db.execute(
        'UPDATE all_data SET ml_assigned = 1, ml_probability = RANDOM() % 100 / 100.0 WHERE ml_assigned = 0'
    )
    db.commit()

    return redirect(url_for('machine_learning.index'))