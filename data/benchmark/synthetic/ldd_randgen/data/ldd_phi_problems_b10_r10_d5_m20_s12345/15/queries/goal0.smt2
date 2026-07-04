(set-logic QF_IDL)
(declare-fun x1 () Int)
(assert (let ((.def_0 (<= x1 1))) .def_0))
(check-sat)
