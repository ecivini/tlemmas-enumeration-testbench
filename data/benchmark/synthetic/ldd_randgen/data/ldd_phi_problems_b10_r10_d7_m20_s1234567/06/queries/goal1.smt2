(set-logic QF_IDL)
(declare-fun x3 () Int)
(assert (let ((.def_0 (<= x3 1))) .def_0))
(check-sat)
