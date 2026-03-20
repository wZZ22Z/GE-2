#include "common/pybind_headers.h"

void init_gege(py::module &);

PYBIND11_MODULE(_manager, m) {
    m.doc() = "High level execution management.";

    // manager
    init_gege(m);
}
