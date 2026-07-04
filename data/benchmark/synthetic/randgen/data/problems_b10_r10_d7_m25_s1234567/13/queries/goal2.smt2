(set-logic QF_RDL)
(declare-fun x3 () Real)
(assert (let ((.def_0 (<= (- (/ 2262688069640987.0 2651193592250910.0)) x3))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
