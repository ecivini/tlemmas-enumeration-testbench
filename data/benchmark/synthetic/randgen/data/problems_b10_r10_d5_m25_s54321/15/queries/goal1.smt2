(set-logic QF_UF)
(declare-fun A3 () Bool)
(assert (let ((.def_0 (not A3))) .def_0))
(check-sat)
