(set-logic QF_RDL)
(declare-fun x2 () Real)
(assert (let ((.def_0 (<= (- (/ 3319223716754736.0 2864392562756803.0)) x2))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
