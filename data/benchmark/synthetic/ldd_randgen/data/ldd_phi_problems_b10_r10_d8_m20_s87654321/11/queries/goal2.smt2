(set-logic QF_IDL)
(declare-fun x0 () Int)
(assert (let ((.def_0 (<= (- 1) x0))) .def_0))
(check-sat)
