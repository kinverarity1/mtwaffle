'''Setup script for mtwaffle'''
from setuptools import setup

setup(
    name='mtwaffle',
    version='0.4',
    description='Magnetotelluric data analysis',
    long_description=open('README.md').read(),
    url='https://github.com/kinverarity1/mtwaffle',
    author='Kent Inverarity',
    author_email='kinverarity@hotmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Customer Service',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Filesystems',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='science geophysics',
    packages=['mtwaffle', ],
    install_requires=['numpy', 'scipy', 'matplotlib', 'attrdict'],
)
