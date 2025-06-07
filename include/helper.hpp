#ifndef HELPER_H_   
#define HELPER_H_

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define MAX_CHAR_ARRAY 1000

/* Calculates nCk, i.e. number of combinations from 'n' items taken 'k' at a time.
 * Used to calculate performance metric, i.e. number of unique sets (i.e. combinations) of SNPs processed per second scaled to sample size. */
unsigned long long n_choose_k(unsigned int n, unsigned int k);

#endif
