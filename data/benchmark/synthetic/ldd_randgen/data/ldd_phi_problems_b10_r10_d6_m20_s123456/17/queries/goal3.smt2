(set-logic QF_UF)
(declare-fun A2 () Bool)
(assert (let ((.def_0 (not A2))) .def_0))
(check-sat)
