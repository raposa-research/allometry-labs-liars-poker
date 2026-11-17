#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "lp_tree.h"

namespace py = pybind11;
using lp::Real;

namespace {

struct PyTraverser {
  std::unique_ptr<lp::Traverser> t;
  // layout of the pending chunk
  lp::Chunk chunk;
  std::vector<uint8_t> infostates, legal;

  explicit PyTraverser(int num_digits, int hand_length)
      : t(new lp::Traverser(lp::Config{num_digits, hand_length})) {}

  int NH() const { return t->config().num_hands(); }
  int NA() const { return t->config().num_actions(); }
  int IS() const { return t->config().infostate_size(); }

  py::object next_chunk() {
    bool ok;
    {
      py::gil_scoped_release release;
      ok = t->next_chunk(chunk, infostates, legal);
    }
    if (!ok) return py::none();
    const py::ssize_t rows = (py::ssize_t)(chunk.end - chunk.begin) * NH();
    py::array_t<uint8_t> is({rows, (py::ssize_t)IS()}, infostates.data());
    py::array_t<bool> lg({rows, (py::ssize_t)NA()},
                         reinterpret_cast<const bool *>(legal.data()));
    return py::make_tuple(chunk.player, is, lg);
  }

  void push_probs(py::array_t<Real, py::array::c_style | py::array::forcecast> probs) {
    const py::ssize_t rows = (py::ssize_t)(chunk.end - chunk.begin) * NH();
    if (probs.ndim() != 2 || probs.shape(0) != rows ||
        (probs.shape(1) != NA() && probs.shape(1) != NA() + 1))
      throw std::invalid_argument("probs must have shape (" + std::to_string(rows) +
                                  ", " + std::to_string(NA()) + ") or (" + std::to_string(rows) +
                                  ", " + std::to_string(NA() + 1) + ") with a trailing lose column");
    const size_t stride = (size_t)probs.shape(1);
    const Real *lose = stride == (size_t)NA() + 1 ? probs.data() + NA() : nullptr;
    py::gil_scoped_release release;
    t->push_probs(probs.data(), stride, lose, stride);
  }

  py::dict finish(bool store_br) {
    lp::Result r;
    {
      py::gil_scoped_release release;
      r = t->finish(store_br);
    }
    py::dict out;
    out["ev0"] = r.ev0;
    out["expl"] = py::make_tuple(r.expl[0], r.expl[1]);
    out["br_value"] = py::make_tuple(r.br_value[0], r.br_value[1]);
    return out;
  }

  // vectorised navigation helpers for simulations: all nodes at one level
  py::array_t<int64_t> child_indices(int level, py::array_t<int64_t> idx, py::array_t<int64_t> action) {
    if (idx.ndim() != 1 || action.ndim() != 1 || idx.shape(0) != action.shape(0))
      throw std::invalid_argument("idx and action must be 1-d arrays of equal length");
    py::array_t<int64_t> out(idx.shape(0));
    auto i = idx.unchecked<1>(), a = action.unchecked<1>();
    auto o = out.mutable_unchecked<1>();
    for (py::ssize_t k = 0; k < idx.shape(0); ++k)
      o(k) = t->child_index(level, (uint32_t)i(k), (int)a(k));
    return out;
  }

  py::array_t<int64_t> br_actions(int level, py::array_t<int64_t> idx, py::array_t<int64_t> hand) {
    if (idx.ndim() != 1 || hand.ndim() != 1 || idx.shape(0) != hand.shape(0))
      throw std::invalid_argument("idx and hand must be 1-d arrays of equal length");
    py::array_t<int64_t> out(idx.shape(0));
    auto i = idx.unchecked<1>(), h = hand.unchecked<1>();
    auto o = out.mutable_unchecked<1>();
    for (py::ssize_t k = 0; k < idx.shape(0); ++k)
      o(k) = t->br_action(level, (uint32_t)i(k), (int)h(k));
    return out;
  }

  py::tuple node_infostate(int level, uint32_t idx, int hand) {
    const lp::PublicState s = t->replay(level, idx);
    py::array_t<uint8_t> is(IS());
    py::array_t<bool> lg(NA());
    t->infostate_tensor(s, hand, is.mutable_data());
    t->legal_mask(s, reinterpret_cast<uint8_t *>(lg.mutable_data()));
    return py::make_tuple(is, lg);
  }
};

}  // namespace

PYBIND11_MODULE(pylpeas, m) {
  m.doc() = "Exact exploitability for 2-player Liar's Poker (OpenSpiel rules)";

  py::class_<PyTraverser>(m, "LiarsPokerTraverser")
      .def(py::init<int, int>(), py::arg("num_digits"), py::arg("hand_length"))
      .def_property_readonly("num_digits", [](const PyTraverser &s) { return s.t->config().num_digits; })
      .def_property_readonly("hand_length", [](const PyTraverser &s) { return s.t->config().hand_length; })
      .def_property_readonly("num_hands", &PyTraverser::NH)
      .def_property_readonly("num_actions", &PyTraverser::NA)
      .def_property_readonly("infostate_size", &PyTraverser::IS)
      .def_property_readonly("num_levels", [](const PyTraverser &s) { return s.t->num_levels(); })
      .def_property_readonly("num_decision_nodes", [](const PyTraverser &s) { return s.t->num_decision_nodes(); })
      .def_property_readonly("num_terminals", [](const PyTraverser &s) { return s.t->num_terminals(); })
      .def("level_size", [](const PyTraverser &s, int L) { return s.t->level_size(L); }, py::arg("level"))
      .def("level_player", [](const PyTraverser &, int L) { return lp::Traverser::level_player(L); }, py::arg("level"))
      .def("num_infosets", [](const PyTraverser &s, int p) { return s.t->num_infosets(p); }, py::arg("player"))
      .def("infostate_string", [](const PyTraverser &s, int L, uint32_t i, int h) { return s.t->infostate_string(L, i, h); },
           py::arg("level"), py::arg("idx"), py::arg("hand"))
      .def("node_infostate", &PyTraverser::node_infostate, py::arg("level"), py::arg("idx"), py::arg("hand"),
           "(information_state_tensor, legal_mask) of a (node, hand) pair")
      .def("begin", [](PyTraverser &s, uint32_t chunk_nodes) { s.t->begin(chunk_nodes); }, py::arg("chunk_nodes") = 20000)
      .def("next_chunk", &PyTraverser::next_chunk,
           "None when the top-down pass is complete, else (player, infostates uint8 [n, IS], legal bool [n, NA])")
      .def("push_probs", &PyTraverser::push_probs, py::arg("probs"),
           "probs float64 [n, NA] (rows sum to 1 over legal actions), or [n, NA + 1] whose last column "
           "is the mass on immediately-losing 'dummy' actions (rows sum to 1 including it)")
      .def("finish", &PyTraverser::finish, py::arg("store_br") = false,
           "dict(ev0, expl=(expl0, expl1), br_value=(br0, br1)); store_br keeps the best-response policy")
      .def_property_readonly("has_br_policy", [](const PyTraverser &s) { return s.t->has_br_policy(); })
      .def("child_index", [](const PyTraverser &s, int L, uint32_t i, int a) { return s.t->child_index(L, i, a); },
           py::arg("level"), py::arg("idx"), py::arg("action"), "child node index in level+1, or -1 if terminal/illegal")
      .def("child_indices", &PyTraverser::child_indices, py::arg("level"), py::arg("idx"), py::arg("action"))
      .def("br_action", [](const PyTraverser &s, int L, uint32_t i, int h) { return s.t->br_action(L, i, h); },
           py::arg("level"), py::arg("idx"), py::arg("hand"), "best-response action (after finish(store_br=True))")
      .def("br_actions", &PyTraverser::br_actions, py::arg("level"), py::arg("idx"), py::arg("hand"));
}
