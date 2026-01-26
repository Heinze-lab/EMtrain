from setuptools import setup, find_packages

VERSION = '0.1' 
DESCRIPTION = ''
LONG_DESCRIPTION = ''

# Setting up
setup(
        name="emtrain", 
        version=VERSION,
        author="Valentin Gillet",
        author_email="valentin.gillet@biol.lu.se",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'gunpowder', 'daisy'], 
        keywords=[]
    )
