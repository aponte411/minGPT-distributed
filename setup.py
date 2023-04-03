from setuptools import setup

setup(
    name='minGPT-distributed',
    version='0.0.1',
    author='David Aponte',
    packages=['mingpt'],
    description='A PyTorch re-implementation of GPT trained on multiple nodes',
    license='MIT',
    install_requires=['torch', 'hydra-core'],
)
