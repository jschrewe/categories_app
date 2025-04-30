from flask import Blueprint, render_template, request, redirect, url_for
from .db import get_db

from .random_sample import sample_assigned

# Create the blueprint
bp = Blueprint('machine_learning', __name__, url_prefix='/machine_learning')


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