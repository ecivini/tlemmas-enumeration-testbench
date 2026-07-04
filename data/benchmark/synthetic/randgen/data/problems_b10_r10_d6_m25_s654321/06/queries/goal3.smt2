(set-logic QF_UF)
(declare-fun A1 () Bool)
(assert (let ((.def_0 (not A1))) .def_0))
(check-sat)
