(set-logic QF_IDL)
(declare-fun x0 () Int)
(assert (let ((.def_0 (<= x0 (- 1)))) .def_0))
(check-sat)
