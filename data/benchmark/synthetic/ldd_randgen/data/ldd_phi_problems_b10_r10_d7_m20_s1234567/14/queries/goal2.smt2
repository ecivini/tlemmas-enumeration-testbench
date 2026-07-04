(set-logic QF_IDL)
(declare-fun x4 () Int)
(assert (let ((.def_0 (<= x4 1))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
