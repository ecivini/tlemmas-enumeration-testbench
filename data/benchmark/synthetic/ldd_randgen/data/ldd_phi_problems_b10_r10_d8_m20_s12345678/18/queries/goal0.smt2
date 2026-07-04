(set-logic QF_IDL)
(declare-fun x5 () Int)
(assert (let ((.def_0 (<= x5 (- 1)))) .def_0))
(check-sat)
