#include "common/pybind_headers.h"

#include "gege.h"

void init_gege(py::module &m) {
    m.def("gege_train", &gege_train, py::arg("config"), py::call_guard<py::gil_scoped_release>());
    m.def("gege_eval", &gege_eval, py::arg("config"), py::call_guard<py::gil_scoped_release>());
}
