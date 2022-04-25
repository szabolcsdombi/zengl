import os

import requests
from progress.bar import Bar

project_root = os.path.dirname(os.path.normpath(os.path.abspath(__file__)))

while not os.path.isfile(os.path.join(project_root, 'setup.py')):
    project_root = os.path.dirname(project_root)


def get(filename):
    os.makedirs(os.path.join(project_root, 'downloads'), exist_ok=True)
    full_path = os.path.join(project_root, 'downloads', filename)
    if os.path.isfile(full_path):
        return full_path
    with requests.get(f'https://f003.backblazeb2.com/file/zengl-data/examples/{filename}', stream=True) as request:
        total_size = int(request.headers.get('Content-Length'))
        print(f'Downloading {filename}')
        bar = Bar('Progress', fill='-', suffix='%(percent)d%%', max=total_size)
        with open(full_path + '.temp', 'wb') as f:
            chunk_size = (total_size + 100 - 1) // 100
            for chunk in request.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.next(len(chunk))
        os.rename(full_path + '.temp', full_path)
        bar.finish()
    return full_path
