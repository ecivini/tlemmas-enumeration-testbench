(set-logic QF_IDL)
(declare-fun x0 () Int)
(assert (let ((.def_0 (<= 0 x0))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
