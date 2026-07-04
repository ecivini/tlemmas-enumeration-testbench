(set-logic QF_RDL)
(declare-fun x3 () Real)
(assert (let ((.def_0 (<= x3 1.0))) .def_0))
(check-sat)
