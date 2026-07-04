(set-logic QF_RDL)
(declare-fun x0 () Real)
(assert (let ((.def_0 (<= x0 1.0))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
