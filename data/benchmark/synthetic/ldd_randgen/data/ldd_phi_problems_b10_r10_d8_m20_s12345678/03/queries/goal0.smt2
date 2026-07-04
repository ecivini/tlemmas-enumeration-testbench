(set-logic QF_IDL)
(declare-fun x3 () Int)
(assert (let ((.def_0 (<= 1 x3))) .def_0))
(check-sat)
