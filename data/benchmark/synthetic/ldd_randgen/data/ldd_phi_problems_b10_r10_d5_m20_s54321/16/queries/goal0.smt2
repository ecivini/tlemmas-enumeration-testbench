(set-logic QF_IDL)
(declare-fun x2 () Int)
(assert (let ((.def_0 (<= x2 1))) .def_0))
(check-sat)
