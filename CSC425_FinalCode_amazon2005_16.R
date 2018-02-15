###################################################################
#Name: Brunda Chouthoy
#CSC 425 Time Series Analysis and forecasting
#Amazon.com, Inc. (AMZN) Historical Price data 
#Analysis of Amazon daily stock returns
#Time Period: 2005-2016(Present)
#Depaul ID: 1804455
###################################################################

# Load the libraries
library(rgl)
library(rugarch)
library(tseries)
library(fBasics)
library(zoo)
library(forecast)
library(fBasics)

# import data in R and compute log returns
# import libraries for TS analysis
myd= read.table('amazon_2005_16.csv', header=T, sep=',')
head(myd)
#Create a timeseries object with the zoo function
amazon.ts = zoo(myd$Close, as.Date(as.character(myd$Date), format=c("%m/%d/%y")))
#log return time series
rets = log(amazon.ts/lag(amazon.ts, -1))
# strip off the dates and just create a simple numeric object
ret = coredata(rets);

###################################################################
##EXPLORATORY ANALYSIS OF THE DATA 
###################################################################
#compute statistics
basicStats(rets)
hist(rets, xlab="Stock returns", prob=TRUE, main="Histogram: Amazon stock returns data(2005-16)") 
##add approximating normal density curve 
xfit<-seq(min(rets),max(rets),length=40) 
yfit<-dnorm(xfit,mean=mean(rets),sd=sd(rets)) 
lines(xfit, yfit, col="blue", lwd=2)


# creates time plot of log returns
plot(rets, ylab="Log of stock returns", xlab='Year', main="Time plot:Amazon stock returns data(2005-16)")
#plot returns, square returns and abs(returns)
par(mfrow=c(2,1))
plot(rets, ylab="Log of stock returns", xlab='Year', main="Time plot:Amazon stock returns data(2005-16)")
plot(rets^2,type='l')
plot(abs(rets),type='l')

par(mfrow=c(2,1))
# Plots ACF function of vector data
acf(ret)
# Plot ACF of squared returns to check for ARCH effect 
acf(ret^2)
# Plot ACF of absolute returns to check for ARCH effect 
acf(abs(ret))

par(mfrow=c(1,1))
# Computes Ljung-Box test on returns to test  independence 
Box.test(coredata(rets),lag=6,type='Ljung')
Box.test(coredata(rets),lag=9,type='Ljung')
Box.test(coredata(rets),lag=12,type='Ljung')
# Computes Ljung-Box test on squared returns to test non-linear independence 
Box.test(coredata(rets^2),lag=6,type='Ljung')
Box.test(coredata(rets^2),lag=12,type='Ljung')
# Computes Ljung-Box test on absolute returns to test non-linear independence 
Box.test(abs(coredata(rets)),lag=6,type='Ljung')
Box.test(abs(coredata(rets)),lag=12,type='Ljung')

# plots PACF of squared returns to identify order of AR model
pacf(coredata(rets),lag=20)

###################################################################
##MODEL FITTING, RESIDUAL ANALYSIS AND MODEL DIAGNOSTICS
###################################################################
#specify model using functions in rugarch package
#Fit ARMA(0,0)-GARCH(1,1) model with Normal distribution
garch11.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)))
#estimate model 
garch11.fit=ugarchfit(spec=garch11.spec, data=rets)
garch11.fit
#plot of residuals
#plot(garch11.fit)

#Fit ARMA(0,0)-GARCH(1,1) model with t-distribution
garch11.t.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
garch11.t.fit=ugarchfit(spec=garch11.t.spec, data=rets)
garch11.t.fit
#plot of residuals
plot(garch11.t.fit)

#using extractors
#estimated coefficients:
coef(garch11.fit)
#unconditional mean in mean equation
uncmean(garch11.fit)
#unconditional varaince: omega/(alpha1+beta1)
uncvariance(garch11.fit)
#persistence = alpha1+beta1
persistence(garch11.fit)
#half-life: ln(0.5)/ln(alpha1+beta1)
halflife(garch11.fit)

#Fit ARMA(0,0)-GARCH(1,1) model with skewed t-distribution
garch11.skt.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "sstd")
#estimate model 
garch11.skt.fit=ugarchfit(spec=garch11.skt.spec, data=rets)
garch11.skt.fit
persistence(garch11.skt.fit)

#Fit ARMA(0,0)-eGARCH(1,1) model with  Gaussian distribution
egarch11.spec=ugarchspec(variance.model=list(model = "eGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)))
#estimate model 
egarch11.fit=ugarchfit(spec=egarch11.spec, data=ret)
egarch11.fit

#Fit ARMA(0,0)-eGARCH(1,1) model with t-distribution
egarch11.t.spec=ugarchspec(variance.model=list(model = "eGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
egarch11.t.fit=ugarchfit(spec=egarch11.t.spec, data=rets)
egarch11.t.fit
persistence(egarch11.t.fit)
#plot of residuals
plot(egarch11.t.fit)

#FORECASTS for egarch.t.fit
#compute h-step ahead forecasts for h=1,2,...,10
egarch11.t.fcst=ugarchforecast(egarch11.t.fit, n.ahead=12)
egarch11.t.fcst
plot(egarch11.t.fcst)

#Fit ARMA(0,0)-TGARCH(1,1) model with t-distribution
gjrgarch11.t.spec=ugarchspec(variance.model=list(model = "gjrGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
gjrgarch11.t.fit=ugarchfit(spec=gjrgarch11.t.spec, data=ret)
gjrgarch11.t.fit

#Fit ARMA(0,0)-gjrGARCH(1,1) model with skewed t-distribution
gjrgarch11.t.spec=ugarchspec(variance.model=list(model = "gjrGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "sstd")
#estimate model 
gjrgarch11.t.fit=ugarchfit(spec=gjrgarch11.t.spec, data=ret)
gjrgarch11.t.fit

# MODEL COMPARISON
# compare information criteria
model.list = list(garch11 = garch11.fit, garch11.t = garch11.t.fit,
                  egarch11 = egarch11.t.fit,
                  gjrgarch11 = gjrgarch11.t.fit)

info.mat = sapply(model.list, infocriteria)
rownames(info.mat) = rownames(infocriteria(garch11.fit))
info.mat

###################################################################
##FORECAST ANALYSIS AND BACKTESTING
###################################################################
# re-fit models leaving 300 out-of-sample observations for forecast
# evaluation statistics
egarch11.t.fit = ugarchfit(egarch11.t.spec, data=ret, out.sample=300)

egarch11.t.fcst = ugarchforecast(egarch11.t.fit, n.roll=300, n.ahead=20)
egarch11.t.fcst
plot(egarch11.t.fcst, which = "all")

# compute forecast evaluation statistics using fpm method
# type="fpm" shows forecast performance measures 
# (Mean Squared Error (MSE), mean absolute error(MAE) and directional accuracy 
# of the forecasts vs realized returns(DAC)).
fcst.list = list(garch11.t=garch11.t.fcst,
                 egarch11.t=egarch11.t.fcst)
fpm.mat = sapply(fcst.list, fpm)
fpm.mat


## Backtesting method to compare EGARCH and GARCH models:
mod_egarch = ugarchroll(egarch11.t.spec, data = rets, n.ahead = 1, 
                        n.start = 1600, refit.every = 200, refit.window = "recursive")

mod_garch = ugarchroll(garch11.t.spec, data = rets, n.ahead = 1,
                      n.start = 1600, refit.every = 200, refit.window = "recursive")
report(mod_egarch, type="fpm")
report(mod_garch, type="fpm")

#type=VaR shows VaR at 1% level: this is the tail probablity. 
report(mod_egarch, type = "VaR", VaR.alpha =0.01, conf.level = 0.95)

#to visualize results use plot
plot(mod_egarch)
plot(mod_garch)