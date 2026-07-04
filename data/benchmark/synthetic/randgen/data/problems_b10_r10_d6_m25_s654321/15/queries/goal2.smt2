(set-logic QF_RDL)
(declare-fun x2 () Real)
(assert (let ((.def_0 (<= 0.0 x2))) .def_0))
(check-sat)
