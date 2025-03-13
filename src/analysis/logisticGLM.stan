data {
  int<lower=0> N;  // no. of outcomes
  int<lower=0> P;  // no. of predictor variables
  array[N] int<lower=0, upper=1> y;  // outcomes
  matrix[N, P] X;  // predictor variables
}

parameters {
  vector[N] alpha;
  vector[P] beta;
}

model {
  for (i in 1:N) {
    alpha[i] ~ student_t(7, 0, 2.5);
  }
  
  for (j in 1:P) {
    beta[j] ~ student_t(7, 0, 2.5);
  }
  
  y ~ bernoulli_logit_glm(X, alpha, beta);
}
