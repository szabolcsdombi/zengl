project = 'ZenGL'
copyright = '2023, Szabolcs Dombi'
author = 'Szabolcs Dombi'

release = '1.16.0'

extensions = []
templates_path = []
exclude_patterns = []

html_title = f'ZenGL {release}'
html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    'source_repository': 'https://github.com/szabolcsdombi/zengl/',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'light_logo': 'logo.png',
    'dark_logo': 'logo.png',
}


def setup(app):
    app.add_css_file('css/custom.css')
