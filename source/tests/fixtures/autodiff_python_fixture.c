#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "VernonRuntime.h"

#ifndef VERNON_AUTODIFF_FIXTURE_REGISTRATION
#error "VERNON_AUTODIFF_FIXTURE_REGISTRATION must name the fixture registration function"
#endif
#ifndef VERNON_AUTODIFF_FIXTURE_MODULE
#error "VERNON_AUTODIFF_FIXTURE_MODULE must name the Python module"
#endif

#define VERNON_CONCAT_IMPL(left, right) left##right
#define VERNON_CONCAT(left, right) VERNON_CONCAT_IMPL(left, right)
#define VERNON_STRING_IMPL(value) #value
#define VERNON_STRING(value) VERNON_STRING_IMPL(value)

extern VernonStatus VERNON_AUTODIFF_FIXTURE_REGISTRATION(void);

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, VERNON_STRING(VERNON_AUTODIFF_FIXTURE_MODULE),
                                    "Registers one statically linked cooked autodiff fixture.", -1, NULL};

PyMODINIT_FUNC VERNON_CONCAT(PyInit_, VERNON_AUTODIFF_FIXTURE_MODULE)(void) {
    if (VERNON_AUTODIFF_FIXTURE_REGISTRATION() != VERNON_STATUS_OK) {
        PyErr_SetString(PyExc_RuntimeError, "cannot register autodiff fixture");
        return NULL;
    }
    return PyModule_Create(&module);
}
