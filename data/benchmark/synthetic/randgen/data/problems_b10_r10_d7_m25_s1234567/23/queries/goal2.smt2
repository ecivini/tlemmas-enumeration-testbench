(set-logic QF_RDL)
(declare-fun x3 () Real)
(assert (let ((.def_0 (<= 0.0 x3))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
