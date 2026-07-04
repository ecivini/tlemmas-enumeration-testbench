(set-logic QF_IDL)
(declare-fun x5 () Int)
(assert (let ((.def_0 (<= 1 x5))) .def_0))
(check-sat)
