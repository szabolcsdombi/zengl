project = 'zengl'
copyright = '2022, Szabolcs Dombi'
author = 'Szabolcs Dombi'

release = '1.9.0'

extensions = []
templates_path = []
exclude_patterns = []

html_theme = 'furo'
html_static_path = ['_static']


def setup(app):
    app.add_css_file('css/custom.css')
