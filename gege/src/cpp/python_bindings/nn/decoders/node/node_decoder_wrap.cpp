#include <common/pybind_headers.h>

#include "nn/decoders/node/node_decoder.h"

void init_node_decoder(py::module &m) {
    py::class_<NodeDecoder, Decoder, shared_ptr<NodeDecoder>>(m, "NodeDecoder").def("forward", &NodeDecoder::forward, py::arg("inputs"));
}
