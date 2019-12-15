import os.path
import re
from numpy.distutils.core import setup, Extension


def find_version(*paths):
    fname = os.path.join(os.path.dirname(__file__), *paths)
    with open(fname) as fp:
        code = fp.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", code, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


ext_cpwt = Extension(name='plateflex.cpwt',
                     sources=['src/cpwt/cpwt.f90', 'src/cpwt/cpwt_sub.f90'])
ext_flex = Extension(name='plateflex.flex',
                     sources=['src/flex/flex.f90'])

setup(
    name='plateflex',
    version=find_version('plateflex', '__init__.py'),
    description='Python package for estimating lithospheric elastic thickness',
    author='Pascal Audet',
    maintainer='Pascal Audet',
    author_email='pascal.audet@uottawa.ca',
    url='https://github.com/paudetseis/PlateFlex',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Fortran',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'],
    install_requires=['numpy>=1.15', 'pymc3', 'matplotlib', 'seaborn'],
    python_requires='>=3.5',
    tests_require=['pytest'],
    ext_modules=[ext_cpwt, ext_flex],
    packages=['plateflex'],
    package_data={
        'plateflex': [
            'examples/data.zip',
            'examples/Notebooks/*.ipynb']
    }
)
