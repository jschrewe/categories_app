import os

from flask import Flask, render_template

from . import db as db_mod
from . import random_sample
from . import machine_learning


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(
            app.instance_path,
            'categories.sqlite'
        ),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db_mod.init_app(app)

    # Register the random_sample blueprint
    app.register_blueprint(random_sample.bp)
    app.register_blueprint(machine_learning.bp)

    # a simple page that says hello
    @app.route('/')
    def index():
        db = db_mod.get_db()
        data = db.execute('SELECT * FROM all_data').fetchall()
        return render_template('index.html', data=data)

    @app.teardown_appcontext
    def close_db_connection(exception):
        db_mod.close_db()

    return app