(set-logic QF_IDL)
(declare-fun x7 () Int)
(assert (let ((.def_0 (<= 1 x7))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
