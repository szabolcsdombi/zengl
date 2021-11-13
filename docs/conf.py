project = 'zengl'
copyright = '2021, Szabolcs Dombi'
author = 'Szabolcs Dombi'

release = '1.0.1'

extensions = [
    'sphinx_rtd_theme',
]

templates_path = []
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def setup(app):
    app.add_css_file('css/custom.css')
