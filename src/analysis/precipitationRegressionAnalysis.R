#library(bayesplot)
library(dplyr)
library(ggplot2)
#library(ggmcmc)
library(reshape2)
#library(rstan)
#library(rstanarm)

set.seed(12345)
setwd("~/projects/FORMES/rainfall-pde-ml/src/analysis/")

# Load timeseries data
series <- data.frame(read.csv("~/projects/FORMES/rainfall-pde-ml/data/timeseriesGPM+ERA5.csv"))

# Calculate shear from Walz et al, pg. 7
shr <- series$u600 - series$u925
shr_norm <- shr / sqrt(sum(shr^2))
series$shr <- shr_norm

# Visualize correlations
series <- series %>% select(sort(names(series)))
corr <- cor(series)
melted_cormat <- melt(corr)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + geom_tile()

################################################################################

# Model 0a: stepwise variable selection with lm - only variables from paper
lm0a <- lm(precip ~ cape + cin + crwc300 + crwc500 + crwc600 + crwc700 + 
             crwc850 + crwc925 + crwc950 + d2m + kx + q500 + q600 + q700 + 
             q925 + r300 + r500 + shr + sp + t2m + t500 + t850 + tcc + tclw + 
             tcwv + time1 + time2 + vimd, 
           data=series)

slm1 <- step(lm0a, direction="both")
sink("lm1.txt")  # outputs next line to txt file
print(summary(slm1))

# Model 0b: stepwise variable selection with lm - all variables downloaded
temp_series <- series %>% select(-precip_thresh)
lm0b <- lm(precip ~ ., data=temp_series)

slm2 <- step(lm0b, direction="both")
sink("lm2.txt")  # outputs next line to txt file
print(summary(slm2))

sink()  # returns output to console

# Note: u600, u925, and shr are correlated (see calculation above)
# Note: dataset missing v (north-south wind at levels) - currently downloading
# Note: step() incrementally adds/removes cvariates using AIC (lower is better) 

################################################################################

# Model 1: Binomial GLM
t_prior <- student_t(7, 0, 2.5)
fit_m1 <- stan_glm(precip_thresh ~ cape + cin + crwc300 + crwc500 + cwrc600 + 
                     crwc700 + crwc850 + crwc925 + crwc950 + d2m + kx + 
                     q500 + q600 + q700 + q925 + r300 + r500 + sp + t2m + 
                     t500 + t850 + tcc + tclw + tcwv + time1 + time2 + vimd, 
                   data=res$train, family=binomial(link="logit"), prior=t_prior,
                   cores=2, seed=12345, chains=4)

################################################################################

# Process data into Stan-usable format
predictors <- series %>% select(-precip, -precip_thresh)
y <- series$precip_thresh
N <- nrow(series)
P <- ncol(series) - 2
stan_data <- list(N=N, P=P, y=y, X=predictors)

# Model 2: Logistic GLM
fit_m1 <- stan(file="logisticGLM.stan", data=stan_data)
fit_m1 %>% mcmc_trace()