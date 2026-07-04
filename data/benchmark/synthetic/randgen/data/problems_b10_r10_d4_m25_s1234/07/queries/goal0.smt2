(set-logic QF_RDL)
(declare-fun x1 () Real)
(assert (let ((.def_0 (<= 0.0 x1))) .def_0))
(check-sat)
