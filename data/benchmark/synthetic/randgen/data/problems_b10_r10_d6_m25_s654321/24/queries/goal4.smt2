(set-logic QF_RDL)
(declare-fun x4 () Real)
(assert (let ((.def_0 (<= x4 (/ 3578546418279074.0 1738615759856073.0)))) (let ((.def_1 (<= x4 1.0))) (let ((.def_2 (and .def_1 .def_0))) .def_2))))
(check-sat)
