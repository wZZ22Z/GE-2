#include "common/pybind_headers.h"
#include "engine/evaluator.h"

namespace py = pybind11;

// Trampoline class
class PyEvaluator : Evaluator {
   public:
    using Evaluator::Evaluator;
    void evaluate(bool validation) override { PYBIND11_OVERRIDE_PURE(void, Evaluator, evaluate, validation); }
};

void init_evaluator(py::module &m) {
    py::class_<Evaluator, PyEvaluator, shared_ptr<Evaluator>>(m, "Evaluator")
        .def(py::init<>())
        .def_readwrite("dataloader", &Evaluator::dataloader_)
        .def("evaluate", &Evaluator::evaluate, py::arg("validation"));

    py::class_<SynchronousEvaluator, Evaluator, shared_ptr<SynchronousEvaluator>>(m, "SynchronousEvaluator")
        .def(py::init<shared_ptr<DataLoader>, shared_ptr<Model>>(), py::arg("dataloader"), py::arg("model"));
}
