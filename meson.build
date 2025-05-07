project('plateflex', 'c',
  version : '0.2.0',
  license: 'MIT',
  meson_version: '>=0.64.0',
  default_options : [
    'warning_level=1',
    'c_args=-Wno-unused-function -Wno-conversion -Wno-misleading-indentation -Wno-incompatible-pointer-types',
    'fortran_args=-Wno-conversion',
    'fortran_std=none'],
)

add_languages('fortran', native: false)

#fc = meson.get_compiler('fortran')
#cc = meson.get_compiler('c')

py_mod = import('python')
py = py_mod.find_installation()
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# Unlike distutils, meson doesn't yet include some of the f2py stuff
fortranobject_c = incdir_f2py / 'fortranobject.c'

fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  dependencies: py_dep,
  include_directories: [incdir_numpy, incdir_f2py])

fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: [incdir_numpy, incdir_f2py])


flex_source = custom_target('flexmodule.c',
  input : ['src/flex/flex.f90'],  
  output : ['flexmodule.c', 'flex-f2pywrappers2.f90'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'flex', '--lower']
)
py.extension_module('flex',
  ['src/flex/flex.f90', flex_source],
  include_directories: inc_np, 
  dependencies : [py_dep, fortranobject_dep],
  subdir: 'plateflex',
  install : true
)


cpwt_source = custom_target('cpwtmodule.c',
  input : ['src/cpwt/cpwt_sub.f90', 'src/cpwt/cpwt.f90'], 
  output : ['cpwtmodule.c', 'cpwt-f2pywrappers2.f90'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'cpwt', '--lower']
)
py.extension_module('cpwt',
  [
    'src/cpwt/cpwt_sub.f90',
    'src/cpwt/cpwt.f90', 
    cpwt_source,
  ],
  include_directories: inc_np, 
  dependencies : [py_dep, fortranobject_dep],
  subdir: 'plateflex',
  install : true
)


py.install_sources(
  'plateflex/classes.py',
  'plateflex/conf.py',
  'plateflex/estimate.py',
  'plateflex/plotting.py',
  'plateflex/doc.py',
  'plateflex/__init__.py',
  pure: false,
  subdir: 'plateflex'
  )

install_data(
    'plateflex/examples/data/Bathy_PAC.xyz',
    'plateflex/examples/data/Freeair_PAC.xyz',
    'plateflex/examples/data/crustal_thickness_PAC.xyz',
    'plateflex/examples/data/Topo_NA.xyz',
    'plateflex/examples/data/Freeair_NA.xyz',
    'plateflex/examples/data/Bouguer_NA.xyz',
    'plateflex/examples/data/crustal_density_NA.xyz',
    'plateflex/examples/data/crustal_thickness_NA.xyz',
    install_dir: py.get_install_dir(pure: false) / 'plateflex/examples/data',
)
install_data(
    'plateflex/examples/Notebooks/Ex1_making_grids.ipynb',
    'plateflex/examples/Notebooks/Ex2_wavelet_analysis.ipynb',
    'plateflex/examples/Notebooks/Ex3_admittance_coherence.ipynb',
    'plateflex/examples/Notebooks/Ex4_estimate_flex_parameters_cell.ipynb',
    'plateflex/examples/Notebooks/Ex5_estimate_flex_parameters_grid.ipynb',
    'plateflex/examples/Notebooks/Ex6_full_suite_North_America.ipynb',
    'plateflex/examples/Notebooks/Ex7_full_suite_NW_Pacific.ipynb',
    install_dir: py.get_install_dir(pure: false) / 'plateflex/examples/Notebooks',

)