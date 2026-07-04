(set-logic QF_RDL)
(declare-fun x6 () Real)
(assert (let ((.def_0 (<= x6 1.0))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
