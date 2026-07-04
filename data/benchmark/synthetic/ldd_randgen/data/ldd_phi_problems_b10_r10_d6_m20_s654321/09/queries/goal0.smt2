(set-logic QF_IDL)
(declare-fun x6 () Int)
(assert (let ((.def_0 (<= x6 (- 1)))) .def_0))
(check-sat)
