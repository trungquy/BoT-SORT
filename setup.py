from setuptools import setup, find_packages

def load_requirements(path='requirements.txt', comment_char='#'):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    return [line for line in lines if not line.startswith(comment_char)]

setup(
    name='botsort',
    version='0.2.4',
    maintainer='Ash Hall',
    url='https://github.com/ashwhall/BoT-SORT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=load_requirements('inference-requirements.txt'),
    include_package_data=True
)
