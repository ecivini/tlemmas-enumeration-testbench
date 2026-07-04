(set-logic QF_IDL)
(declare-fun x4 () Int)
(assert (let ((.def_0 (<= 1 x4))) .def_0))
(check-sat)
