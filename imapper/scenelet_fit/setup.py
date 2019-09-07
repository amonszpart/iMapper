from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

# profiling
if False:
    from Cython.Compiler.Options import get_directive_defaults
    directive_defaults = get_directive_defaults()
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True
    define_macros = [('CYTHON_TRACE_NOGIL', '1')]
else:
    directive_defaults = []
    define_macros = []

import numpy
include_dirs = [numpy.get_include()]
gdb_debug = False

extensions = [
    Extension(name="imapper.scenelet_fit.regular_grid_interpolator",
              sources=["imapper/scenelet_fit/regular_grid_interpolator.pyx"],
              include_dirs=include_dirs,
              define_macros=define_macros
              )
    # Extension(name="stealth.scenelet_fit.placement_optimizer_cpu",
    #           sources=["stealth/scenelet_fit/placement_optimizer_cpu.pyx"],
    #           include_dirs=include_dirs,
    #           define_macros=define_macros
    #           ),
    # Extension(name="stealth.scenelet_fit.cost_fun",
    #           sources=["stealth/scenelet_fit/cost_fun.pyx"],
    #           include_dirs=include_dirs,
    #           define_macros=define_macros
    #           ),
    # Extension(name="stealth.scenelet_fit.model_manager",
    #           sources=["stealth/scenelet_fit/model_manager.pyx"],
    #           include_dirs=include_dirs,
    #           define_macros=define_macros
    #           ),
    # Extension(name="stealth.scenelet_fit.unknowns_manager",
    #           sources=["stealth/scenelet_fit/unknowns_manager.pyx"],
    #           include_dirs=include_dirs,
    #           define_macros=define_macros
    #           ),
    # Extension(name="stealth.logic.my_math",
    #           sources=["stealth/logic/my_math.pyx"],
    #           include_dirs=include_dirs,
    #           define_macros=define_macros)
]

setup(
   name='iMapper CPU placement',
   ext_modules=cythonize(extensions, nthreads=4, gdb_debug=gdb_debug,
                         compiler_directives=directive_defaults)
)