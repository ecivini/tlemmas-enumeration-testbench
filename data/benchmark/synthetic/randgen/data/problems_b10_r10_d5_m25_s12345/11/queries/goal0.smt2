(set-logic QF_RDL)
(declare-fun x2 () Real)
(assert (let ((.def_0 (<= x2 1.0))) .def_0))
(check-sat)
