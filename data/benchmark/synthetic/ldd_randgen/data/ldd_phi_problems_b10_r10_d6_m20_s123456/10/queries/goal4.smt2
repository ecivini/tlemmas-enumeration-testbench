(set-logic QF_IDL)
(declare-fun x3 () Int)
(assert (let ((.def_0 (<= 1 x3))) (let ((.def_1 (<= 0 x3))) (let ((.def_2 (and .def_1 .def_0))) .def_2))))
(check-sat)
