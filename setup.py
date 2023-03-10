from setuptools import setup, find_packages

setup(
    name='pandamachine',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas'
    ],
    extras_require={
        'dev': [
            'pytest',
            'coverage',
        ]
    },
    package_data={
        'my_package': ['data/*.csv']
    },
    entry_points={
        'console_scripts': [
            'my_script = my_package.scripts.script:main'
        ]
    },
    author='Mohamed Traore',
    author_email='mt.db@icloud.com',
    description='Panda Machine Description',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/johndoe/my_package',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
