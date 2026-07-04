(set-logic QF_IDL)
(declare-fun x2 () Int)
(assert (let ((.def_0 (<= x2 1))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
