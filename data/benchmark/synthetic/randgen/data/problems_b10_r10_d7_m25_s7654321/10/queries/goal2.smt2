(set-logic QF_RDL)
(declare-fun x2 () Real)
(assert (let ((.def_0 (<= x2 (/ 127708965342824.0 1511063490367039.0)))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
