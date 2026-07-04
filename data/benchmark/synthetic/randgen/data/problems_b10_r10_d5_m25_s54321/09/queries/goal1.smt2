(set-logic QF_RDL)
(declare-fun x1 () Real)
(assert (let ((.def_0 (<= x1 1.0))) .def_0))
(check-sat)
