(set-logic QF_IDL)
(declare-fun x7 () Int)
(assert (let ((.def_0 (<= x7 (- 1)))) .def_0))
(check-sat)
