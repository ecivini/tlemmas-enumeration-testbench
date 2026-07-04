(set-logic QF_IDL)
(declare-fun x8 () Int)
(assert (let ((.def_0 (<= x8 1))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
