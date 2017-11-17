# load libraries
setwd('/home/barendregt/Analysis/PredictionError/data_output/')

num_timepoints = 110

# read data per time point & run ANOVA
pvals_correct <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=3,dimnames=list(c(), c("TR", "TI", "TR*TI"))))
pvals_incorrect <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=3,dimnames=list(c(), c("TR", "TI", "TR*TI"))))

fvals_correct <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=3,dimnames=list(c(), c("TR", "TI", "TR*TI"))))
fvals_incorrect <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=3,dimnames=list(c(), c("TR", "TI", "TR*TI"))))

pvals_all <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=2,dimnames=list(c(), c("type", "TR*TI*type"))))


for(t in 0:(num_timepoints-1)){
  timepoint_data <- read.csv(paste('pupil_data_t',t,'.csv', sep=''))
  
  rpm_correct <- with(timepoint_data[timepoint_data$type=='correct',],  aov(pupil ~ TR*TI + Error(Sub / (TR*TI))))
  rpm_incorrect <- with(timepoint_data[timepoint_data$type=='incorrect',],  aov(pupil ~ TR*TI + Error(Sub / (TR*TI))))
  # rpm_all <- with(timepoint_data, aov(pupil ~ TR*TI*type + Error(Sub / (TR*TI*type))))
  
  fvals_correct[t,1] <- unlist(summary(rpm_correct))["Error: Sub:TR.F value1"]
  fvals_correct[t,2] <- unlist(summary(rpm_correct))["Error: Sub:TI.F value1"]
  fvals_correct[t,3] <- unlist(summary(rpm_correct))["Error: Sub:TR:TI.F value1"]

  pvals_correct[t,1] <- unlist(summary(rpm_correct))["Error: Sub:TR.Pr(>F)1"]
  pvals_correct[t,2] <- unlist(summary(rpm_correct))["Error: Sub:TI.Pr(>F)1"]
  pvals_correct[t,3] <- unlist(summary(rpm_correct))["Error: Sub:TR:TI.Pr(>F)1"]

  fvals_incorrect[t,1] <- unlist(summary(rpm_incorrect))["Error: Sub:TR.F value1"]
  fvals_incorrect[t,2] <- unlist(summary(rpm_incorrect))["Error: Sub:TI.F value1"]
  fvals_incorrect[t,3] <- unlist(summary(rpm_incorrect))["Error: Sub:TR:TI.F value1"]
  # 
  pvals_incorrect[t,1] <- unlist(summary(rpm_incorrect))["Error: Sub:TR.Pr(>F)1"]
  pvals_incorrect[t,2] <- unlist(summary(rpm_incorrect))["Error: Sub:TI.Pr(>F)1"]
  pvals_incorrect[t,3] <- unlist(summary(rpm_incorrect))["Error: Sub:TR:TI.Pr(>F)1"]
  
  # pvals_all[t,1] <- unlist(summary(rpm_all))["Error: Sub:type.F value1"]
  # pvals_all[t,2] <- unlist(summary(rpm_all))["Error: Sub:TR:TI:type.F value1"]  
  
}

# Plot results
par(mfrow=c(2,1))

# CORRECT TRIALS
plot_range = 21:100
fig_range = range(0, fvals_correct$TR[plot_range], fvals_incorrect$TR[plot_range])#, pvals_all$type[plot_range])


# p_value = (vector(length=num_timepoints)+1)*-log(0.05)
# plot(p_value[plot_range], type='l', col='grey', ylab='F', xlab='Time(s)', ylim=fig_range, axes=FALSE)

#axis(1, at=seq(0,num_timepoints,by=5), lab=seq(-1.5,4.5,by=.5))
# axis(2, at=2*0:fig_range[2], las=1)

plot(fvals_correct$TR[plot_range], type='l', col='red', ylab='F-value', xlab ='Time(s)', ylim=fig_range, axes=FALSE)
points(seq(1,length(plot_range)+1)[pvals_correct$TR[plot_range]<0.05],fvals_correct$TR[plot_range][pvals_correct$TR[plot_range]<0.05],type='p',cex=2,pch='*',col='red')
lines(fvals_correct$TI[plot_range], lty='solid',ylim=fig_range, col='green')
points(seq(1,length(plot_range)+1)[pvals_correct$TI[plot_range]<0.05],fvals_correct$TI[plot_range][pvals_correct$TI[plot_range]<0.05],type='p',cex=2,pch='*',col='green')
lines(fvals_correct$'TR*TI'[plot_range], lty='solid',ylim=fig_range, col='black',lw=2)
points(seq(1,length(plot_range)+1)[pvals_correct$'TR*TI'[plot_range]<0.05],fvals_correct$'TR*TI'[plot_range][pvals_correct$'TR*TI'[plot_range]<0.05],type='p',cex=2,pch='*',col='black')

axis(2, at=2*0:fig_range[2], las=1)
axis(1, at=seq(0,100,by=10), lab=seq(0.0,5.0,by=.5))

legend(0, fig_range[2], c("TR","TI",'TR*TI'), cex=0.8, col=c("red",'green','black'),lty=c('solid','solid','solid'),lw=c(1,1,2))
title('Correct')

# INCORRECT TRIALS

# fig_range = range(0, pvals_incorrect$TR[plot_range])

# p_value = (vector(length=num_timepoints)+1)*-log(0.05)
# plot(p_value[plot_range], type='l', col='grey', ylab='F', xlab='Time(s)', ylim=fig_range, axes=FALSE)

#axis(1, at=seq(0,num_timepoints,by=5), lab=seq(-1.5,4.5,by=.5))
# axis(2, at=2*0:fig_range[2], las=1)

plot(fvals_incorrect$TR[plot_range], type='l', col='red', ylab='F-value', xlab ='Time(s)', ylim=fig_range, axes=FALSE)
points(seq(1,length(plot_range)+1)[pvals_incorrect$TR[plot_range]<0.05],fvals_incorrect$TR[plot_range][pvals_incorrect$TR[plot_range]<0.05],type='p',cex=2,pch='*',col='red')
lines(fvals_incorrect$TI[plot_range], lty='solid',ylim=fig_range, col='green')
points(seq(1,length(plot_range)+1)[pvals_incorrect$TI[plot_range]<0.05],fvals_incorrect$TI[plot_range][pvals_incorrect$TI[plot_range]<0.05],type='p',cex=2,pch='*',col='green')
lines(fvals_incorrect$'TR*TI'[plot_range], lty='solid',ylim=fig_range, col='black',lw=2)
points(seq(1,length(plot_range)+1)[pvals_incorrect$'TR*TI'[plot_range]<0.05],fvals_incorrect$'TR*TI'[plot_range][pvals_incorrect$'TR*TI'[plot_range]<0.05],type='p',cex=2,pch='*',col='black')


axis(2, at=2*0:fig_range[2], las=1)
axis(1, at=seq(0,100,by=10), lab=seq(0.0,5.0,by=.5))

legend(0, fig_range[2], c("TR","TI",'TR*TI'), cex=0.8, col=c("red",'green','black'), pch='-', lty=c('solid','solid','solid'),lw=c(1,1,2))
title('Incorrect')


# COMBINED TRIALS

# fig_range = range(0, pvals_all$type[plot_range])

# p_value = (vector(length=num_timepoints)+1)*-log(0.05)Baby Blog Brainstorm
# 
# plot(p_value[plot_range], type='l', col='grey', ylab='-log(p)', xlab='Time(s)', ylim=fig_range, axes=FALSE)


# 

# plot(pvals_all$type[plot_range], type='l', col='blue', ylab='F-value', ylim=fig_range,axes=FALSE)
# # par(new=TRUE)
# lines(pvals_all$'TR*TI*type'[plot_range],  col='black',ylim=fig_range, ylab='')
# 
# axis(1, at=seq(0,100,by=10), lab=seq(0.0,5.0,by=.5))
# axis(2, at=2*0:fig_range[2], las=1)
# 
# legend(0, fig_range[2], c("Trial type",'TR*TI*type'), cex=0.8, col=c("blue",'black'), pch='-', lty='solid')
# title('Combined')