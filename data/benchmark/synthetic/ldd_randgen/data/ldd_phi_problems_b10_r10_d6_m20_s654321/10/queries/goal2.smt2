(set-logic QF_IDL)
(declare-fun x4 () Int)
(assert (let ((.def_0 (<= x4 (- 1)))) .def_0))
(check-sat)
