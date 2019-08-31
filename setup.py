from setuptools import setup

setup(name='xzgen',
      version='0.1.0',
      packages=['xzgen'],
      entry_points={
          'console_scripts': [
              'xzgen = xzgen.__main__:main'
          ]
      },
      )
