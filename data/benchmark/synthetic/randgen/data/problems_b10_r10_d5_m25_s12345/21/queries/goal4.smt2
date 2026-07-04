(set-logic QF_RDL)
(declare-fun x8 () Real)
(assert (let ((.def_0 (<= 0.0 x8))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
