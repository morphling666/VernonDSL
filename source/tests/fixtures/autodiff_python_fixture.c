#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "VernonRuntime.h"

extern VernonStatus vernonRegisterAutodiffFixture(void);

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "vernon_autodiff_fixture", "Registers statically linked cooked autodiff objects.", -1, NULL,
};

PyMODINIT_FUNC PyInit_vernon_autodiff_fixture(void) {
    if (vernonRegisterAutodiffFixture() != VERNON_STATUS_OK) {
        PyErr_SetString(PyExc_RuntimeError, "cannot register cooked autodiff fixture");
        return NULL;
    }
    return PyModule_Create(&module);
}
