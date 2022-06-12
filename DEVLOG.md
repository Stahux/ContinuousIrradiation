Initial notes:
Version v0.1 is taken from my "usual" code, next tasks are to clean the code, add comments/documentation and improve style/readability.
Also the idea is to add following fnctionality (starting from most urgent):
1) Add penalty to selected parameters (to allow them to float within their errors without hard fixing).
2) Add tools to estimate errors (investigate chi2 surface, do bootstrap etc.)
3) Add tools to search for global minimum in chi2
4) Improve graphical model editor to easily define and modify models.
5) Add other chemical processes (like addition, substitution etc.)
6) Optimize fitting function, add multithreades solving of kinetics within single residual iteration.
7) Add simple graphical interface, and some parameter editor, with self-documentation.
...

07.09.2021
I am adding some comments/documentation to old code. Also improving style.
I don't like python style in general (like not using brackets).
Camel case for classes and methods: SomeClass and someClassMethod (like in qt).
With underscore for variables, and protected style for internal class variables: some_variable, _internal_class_variable.
For text "text" not 'text', because it hurts my heart to see more than one letter between ''.
