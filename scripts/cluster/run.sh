#!/bin/bash
/usr/local/sv/svsolver/2019-02-07/svpre model_sim.svpre

/usr/local/sv/svsolver/2019-02-07/svsolver solver.inp

/usr/local/sv/svsolver/2019-02-07/svpost -start 0 -stop 700 -incr 10 -vtp all_results.vtp -vtu all_results.vtu -vtkcombo -all
