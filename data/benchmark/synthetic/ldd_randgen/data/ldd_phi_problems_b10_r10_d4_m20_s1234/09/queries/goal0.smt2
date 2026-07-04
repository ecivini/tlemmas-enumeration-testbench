(set-logic QF_IDL)
(declare-fun x2 () Int)
(assert (let ((.def_0 (<= 0 x2))) .def_0))
(check-sat)
