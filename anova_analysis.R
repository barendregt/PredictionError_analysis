# load libraries
library(rhdf5)
library(lme4)

setwd('/home/barendregt/Analysis/PredictionError')

# read data per time point & run ANOVA
pvals <- as.data.frame(matrix(0,nrow=60,ncol=5,dimnames=list(c(), c("Pupil_PE", "Pupil_TR", "Pupil_correct", "Pupil_incorrect", "RT_con"))))

for(t in 0:59){
  timepoint_data <- as.data.frame(h5read('button_timepoints_matrix.h5',paste('t',t,'/table', sep='')))
  
  pvals[t,1] <- -log(unlist(summary(aov(pupil ~ PE+TR + Error(subject), data=timepoint_data))[[2]])["Pr(>F)1"])
  pvals[t,2] <- -log(unlist(summary(aov(pupil ~ PE+TR + Error(subject), data=timepoint_data))[[2]])["Pr(>F)2"])
  pvals[t,3] <- -log(unlist(summary(aov(pupil ~ PE:TR + Error(subject), data=timepoint_data[timepoint_data$correct==1,]))[[2]])["Pr(>F)1"])
  pvals[t,4] <- -log(unlist(summary(aov(pupil ~ PE:TR + Error(subject), data=timepoint_data[timepoint_data$correct==0,]))[[2]])["Pr(>F)1"])
  pvals[t,5] <- -log(unlist(summary(aov(pupil ~ RT*condition + Error(subject), data=timepoint_data))[[2]])["Pr(>F)3"])
  # pvals[t,5] <- -log(unlist(summary(aov(RT ~ PE+TR + Error(subject), data=timepoint_data))[[2]])["Pr(>F)2"])
  # pvals[t,6] <- -log(unlist(summary(aov(RT ~ PE:TR + Error(subject), data=timepoint_data))[[2]])["Pr(>F)1"])

  H5close()
}

# Plot results
fig_range = range(0, pvals)

p_value = (vector(length=60)+1)*-log(0.05)
plot(p_value, type='l', col='grey', ylab='-log(p)', xlab='Time(s)', ylim=fig_range, axes=FALSE)

axis(1, at=seq(0,60,by=5), lab=seq(-1.5,4.5,by=.5))
axis(2, at=2*0:fig_range[2], las=1)

lines(pvals$Pupil_PE, type='o', col='red')
lines(pvals$Pupil_TR, type='o', col='blue')
lines(pvals$Pupil_correct, type='o', col='black')
lines(pvals$Pupil_incorrect, type='o', col='black',lty='dashed')
lines(pvals$RT_con, type='o', col='grey')

legend(0, fig_range[2], c("Pupil_PE", "Pupil_TR", "Pupil_correct", "Pupil_incorrect", "RT_con"), cex=0.8, col=c("red",'blue','black','black','grey'), pch='o', lty='solid')
title('Button press')


pvals <- as.data.frame(matrix(0,nrow=60,ncol=5,dimnames=list(c(), c("Pupil_PE", "Pupil_TR", "Pupil_correct", "Pupil_incorrect", "RT_con"))))

for(t in 0:59){
  timepoint_data <- as.data.frame(h5read('stim_timepoints_matrix.h5',paste('t',t,'/table', sep='')))
  
  pvals[t,1] <- -log(unlist(summary(aov(pupil ~ PE+TR, data=timepoint_data))[[1]])["Pr(>F)1"])
  pvals[t,2] <- -log(unlist(summary(aov(pupil ~ PE+TR, data=timepoint_data))[[1]])["Pr(>F)2"])
  pvals[t,3] <- -log(unlist(summary(aov(pupil ~ PE:TR, data=timepoint_data[timepoint_data$correct==1,]))[[1]])["Pr(>F)1"])
  pvals[t,4] <- -log(unlist(summary(aov(pupil ~ PE:TR, data=timepoint_data[timepoint_data$correct==0,]))[[1]])["Pr(>F)1"])
  pvals[t,5] <- -log(unlist(summary(aov(pupil ~ RT*condition, data=timepoint_data))[[1]])["Pr(>F)3"])
  # pvals[t,5] <- -log(unlist(summary(aov(RT ~ PE+TR + Error(subject), data=timepoint_data))[[2]])["Pr(>F)2"])
  # pvals[t,6] <- -log(unlist(summary(aov(RT ~ PE:TR + Error(subject), data=timepoint_data))[[2]])["Pr(>F)1"])
  
  H5close()
}

# Plot results
fig_range = range(0, pvals)

p_value = (vector(length=60)+1)*-log(0.05)
plot(p_value, type='l', col='grey', ylab='-log(p)', xlab='Time(s)', ylim=fig_range, axes=FALSE)

axis(1, at=seq(0,60,by=5), lab=seq(-1.5,4.5,by=.5))
axis(2, at=2*0:fig_range[2], las=1)

lines(pvals$Pupil_PE, type='o', col='red')
lines(pvals$Pupil_TR, type='o', col='blue')
lines(pvals$Pupil_correct, type='o', col='black')
lines(pvals$Pupil_incorrect, type='o', col='black',lty='dashed')
lines(pvals$RT_con, type='o', col='grey')
title('Stimulus presentation')
legend(0, fig_range[2], c("Pupil_PE", "Pupil_TR", "Pupil_correct", "Pupil_incorrect", "RT_con"), cex=0.8, col=c("red",'blue','black','black','grey'), pch='o', lty='solid')

