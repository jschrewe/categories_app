import pandas as pd
from flask import Blueprint, render_template, redirect, url_for, request
from .db import get_db

# Create the blueprint
bp = Blueprint('random_sample', __name__, url_prefix='/random_sample')


@bp.route('/')
def index():
    db = get_db()

    data = db.execute(
        'SELECT * FROM all_data WHERE random_sampled = 1 AND assigned = 0'
    ).fetchall()

    # Count total rows in the random sample
    total_count = db.execute(
        'SELECT COUNT(*) FROM all_data WHERE random_sampled = 1'
    ).fetchone()[0]

    # Count rows without a category
    unassigned_count = db.execute(
        'SELECT COUNT(*) FROM all_data WHERE random_sampled = 1 AND category IS NULL'
    ).fetchone()[0]

    return render_template(
        'random_sample/index.html',
        data=data,
        total_count=total_count,
        unassigned_count=unassigned_count
    )


@bp.route('/generate')
def generate_random_sample():
    db = get_db()

    # Clear previous random samples
    db.execute('UPDATE all_data SET random_sampled = 0')

    # Load the data into a Pandas DataFrame
    cursor = db.execute(
        'SELECT * FROM all_data WHERE assigned = 0 AND category IS NULL'
    )
    rows = cursor.fetchall()

    df = pd.DataFrame(
        rows,
        columns=[col[0] for col in cursor.description]
    )

    sampled_df = df.sample(frac=0.2, random_state=1)  # Sample 20% of the data

    # Mark the selected rows as random_sampled in the database
    db.executemany(
        'UPDATE all_data SET random_sampled = 1 WHERE id = ?',
        [(row['id'],) for _, row in sampled_df.iterrows()]
    )

    db.commit()

    # Redirect to the index page to display the new random sample
    return redirect(url_for('random_sample.index'))


@bp.route('/assign', methods=['GET', 'POST'])
def assign_category():
    db = get_db()

    # Count total rows in the random sample
    total_count = db.execute(
        'SELECT COUNT(*) FROM all_data WHERE random_sampled = 1'
    ).fetchone()[0]

    # Count rows without a category
    unassigned_count = db.execute(
        'SELECT COUNT(*) FROM all_data WHERE random_sampled = 1 AND category IS NULL'
    ).fetchone()[0]

    # Fetch the first unassigned entry from the random sample
    entry = db.execute(
        'SELECT * FROM all_data WHERE random_sampled = 1 AND assigned = 0 AND category IS NULL LIMIT 1'
    ).fetchone()

    # If no unassigned entry exists, redirect to the index page
    if not entry:
        return redirect(url_for('random_sample.index'))

    # Fetch all available categories
    categories = db.execute('SELECT * FROM categories').fetchall()

    if request.method == 'POST':
        # Get the selected category from the form
        selected_category = request.form.get('category')

        # Update the entry with the selected category
        db.execute(
            'UPDATE all_data SET category = ?, assigned = 1 WHERE id = ?',
            (selected_category, entry['id'])
        )
        db.commit()

        # Redirect to the same view to assign the next entry
        return redirect(url_for('random_sample.assign_category'))

    # Render the template with the entry, categories, and counts
    return render_template(
        'random_sample/assign.html',
        entry=entry,
        categories=categories,
        total_count=total_count,
        unassigned_count=unassigned_count
    )


def sample_assigned():
    """
    Check if all entries in the random sample have been assigned a category.
    """
    db = get_db()

    # Count total rows in the random sample
    total_count = db.execute(
        'SELECT COUNT(*) FROM all_data WHERE random_sampled = 1'
    ).fetchone()[0]

    # Count rows without a category
    unassigned_count = db.execute(
        'SELECT COUNT(*) FROM all_data WHERE random_sampled = 1 AND category IS NULL'
    ).fetchone()[0]

    if total_count > 0 and unassigned_count == 0:
        return True
    else:
        return False
