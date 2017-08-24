# load libraries
library(rhdf5)
library(lme4)

setwd('/home/barendregt/Projects/PredictionError/Psychophysics/Data/timepoint_data')

# read data per time point & run ANOVA
pvals <- as.data.frame(matrix(0,nrow=54,ncol=4,dimnames=list(c(), c("int", "PU", "UP", "UU"))))

for(t in 0:54){
  timepoint_data <- as.data.frame(h5read('pupil_task_data_diff.h5',paste('t',t,'/table', sep='')))
  
  timepoint_data$correct <- as.factor(timepoint_data$correct)
  timepoint_data$pe_type <- as.factor(timepoint_data$pe_type)
  
  
  
  # pvals[t,1] <- t.test(timepoint_data[])
  
  pvals[t,2] <- t.test(timepoint_data[timepoint_data$pe_type=="PU",]$pupil ~ timepoint_data[timepoint_data$pe_type=="PU",]$correct)[3]
  pvals[t,3] <- t.test(timepoint_data[timepoint_data$pe_type=="UP",]$pupil ~ timepoint_data[timepoint_data$pe_type=="UP",]$correct)[3]
  pvals[t,4] <- t.test(timepoint_data[timepoint_data$pe_type=="UU",]$pupil ~ timepoint_data[timepoint_data$pe_type=="UU",]$correct)[3]

  
  # mdl <- glm(formula = correct ~ pupil*pe_type, family = binomial(link = "logit"), data = timepoint_data)  
  # 
  # pvals[t,1] <- anova(mdl, test="Chisq")["Pr(>Chi)"][4,1]
  # 
  # mdl <- glm(formula = correct ~ pupil, family = binomial(link="logit"), data=timepoint_data[timepoint_data$pe_type=="PU",])
  # 
  # pvals[t,2] <- anova(mdl, test="Chisq")["Pr(>Chi)"][2,1]
  # 
  # mdl <- glm(formula = correct ~ pupil, family = binomial(link="logit"), data=timepoint_data[timepoint_data$pe_type=="UP",])
  # 
  # pvals[t,3] <- anova(mdl, test="Chisq")["Pr(>Chi)"][2,1]
  # 
  # mdl <- glm(formula = correct ~ pupil, family = binomial(link="logit"), data=timepoint_data[timepoint_data$pe_type=="UU",])
  # 
  # pvals[t,4] <- anova(mdl, test="Chisq")["Pr(>Chi)"][2,1]
  
  H5close()
}
# 
pvals[pvals>.45] = 0
pvals['UP'][pvals['UP']>0] = .2
pvals['PU'][pvals['PU']>0] = .1
pvals['UU'][pvals['UU']>0] = .3



# Plot results
fig_range = range(0, pvals)

#p_value = (vector(length=55)+1)*0.45#0
#plot(p_value, type='l', col='grey', ylab='Prob', xlab='Time(s)', ylim=fig_range, axes=FALSE)

plot(pvals$PU, type='p', col='blue', ylab='Predicts answer?', xlab='Time(s)', ylim=fig_range, axes=FALSE)

axis(1, at=seq(0,55,by=5), lab=seq(-0.5,5.0,by=.5))
axis(2, at=seq(0,.3,by=.1), lab=c('0','TI','TR','Both'))

#lines(pvals$int, type='o', col='black',lty='dashed')
lines(pvals$PU, type='p', col='blue')
lines(pvals$UP, type='p', col='red')
lines(pvals$UU, type='p', col='green')
#lines(pvals$RT_con, type='o', col='grey')

# legend(0, fig_range[2], c("TI", "TR", "Both"), cex=0.8, col=c('blue','red','green'), pch='o', lty='solid')
# title('Stimulus presentation')

