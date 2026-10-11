#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifndef VERNON_AUTODIFF_FIXTURE_MODULE
#error "VERNON_AUTODIFF_FIXTURE_MODULE must name the Python module"
#endif

#define VERNON_CONCAT_IMPL(left, right) left##right
#define VERNON_CONCAT(left, right) VERNON_CONCAT_IMPL(left, right)
#define VERNON_STRING_IMPL(value) #value
#define VERNON_STRING(value) VERNON_STRING_IMPL(value)

/* Importing the module loads the linked CPU objects, whose static constructors
 * register their entries with the Runtime. */
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, VERNON_STRING(VERNON_AUTODIFF_FIXTURE_MODULE),
                                    "Statically links one cooked CPU Program fixture.", -1, NULL};

PyMODINIT_FUNC VERNON_CONCAT(PyInit_, VERNON_AUTODIFF_FIXTURE_MODULE)(void) { return PyModule_Create(&module); }
