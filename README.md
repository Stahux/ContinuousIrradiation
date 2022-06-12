# ContinuousIrradiation

This code has been created to simulate, fit and model the data obtained during irradiation by constant light source in stationary spectrometer. It allows to define the points where light has been switched off or modified, so model describing the system is still the same, but conditions changed.

Program has graphical module only to build the model of transitions between states and allows to fix and unfix almost any parameter during fitting process of the data, so everything can be perfectly tuned to get the best data description possible. Building model graphically allows to avoid mistakes common when the equations are written manually (especially for complex models).

Current version is still in development and used in our laboratory exclusively. The functionality is tested, however the code is very slow and awaits deep optimization. Anyone is invited to use the package, however please note that we cannot quarantee that code is completely bug-free. It is tested on the specific cases, which we encountered in our lab.

Plans for future:
1) Rewrite the solver to allow more generalized models, like ones with association, dissociation, etc.
2) Optimize the solver, to generate compiled equaitions first, then solve them (should result is great code speedup)
3) Add exceptions and warnings, where the proper coditions are not met
4) Upload testing code
5) Add code to search for global minima and estimate model indeterminacy
6) Write documentation
