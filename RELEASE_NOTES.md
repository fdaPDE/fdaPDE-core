### v 2.0 (xx july 2023; first independent release)
---
First release
* Release numbering started at release number of `fdaPDE`
* Released as independent submodule after restructuring of `fdaPDE` version 1.1-9

**Major stable features**

1. Expression-template based arithmetic support for multivariate scalar, vector and matrix fields.


Follow Google C++ naming convention (https://google.github.io/styleguide/cppguide.html#Naming)

files names are this_is_a_file.cc
type names are ThisIsAType
variable names are this_is_a_variable
function names are this_is_a_function, this_is_a_method
lines should not be longer than 120 chars
macros are ALL_CAPITAL_LETTER
traits goes like this_is_a_trait<T>::type
in the includes, first the stdlib includes, then ours
curly braces must be always be preceded by a space

----
do not use this style 

if()
{
	....
}

use this 
if() {
	...
}

----
do not put implementation of methods, except for one line methods, inside a class declaration
implementation of large methods are reported out-of-class, immediately after the class declaration, and in 
the same .h file
