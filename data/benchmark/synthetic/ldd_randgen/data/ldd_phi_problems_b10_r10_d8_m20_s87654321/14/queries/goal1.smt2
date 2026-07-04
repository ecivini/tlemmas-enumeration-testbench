(set-logic QF_IDL)
(declare-fun x1 () Int)
(assert (let ((.def_0 (<= (- 1) x1))) .def_0))
(check-sat)
