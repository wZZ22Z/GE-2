#include "common/pybind_headers.h"

// engine
void init_evaluator(py::module &);
void init_graph_encoder(py::module &);
void init_trainer(py::module &);

PYBIND11_MODULE(_engine, m) {
    m.doc() = "Training and Evaluation engine.";

    // engine
    init_evaluator(m);
    init_graph_encoder(m);
    init_trainer(m);
}
