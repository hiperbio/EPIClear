
#include <cfloat>
#include <iostream>

#include "helper.hpp"
#include "reduction.hpp"
#include "tensorop.hpp"


unsigned long long n_choose_k(unsigned int n, unsigned int k)
{
	unsigned long long result = 1;		

	/* Calculates nC_{i} from nC_{i-1} */
	for (unsigned int i = 1; i <= k; i++) {	
		result = result * n / i;		
		n = n - 1;
	}

	return result;
}


