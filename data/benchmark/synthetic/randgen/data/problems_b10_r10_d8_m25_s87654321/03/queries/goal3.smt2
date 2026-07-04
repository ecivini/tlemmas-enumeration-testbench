(set-logic QF_RDL)
(declare-fun x1 () Real)
(assert (let ((.def_0 (<= x1 (- (/ 50036255120798.0 2824721678339493.0))))) (let ((.def_1 (not .def_0))) .def_1)))
(check-sat)
