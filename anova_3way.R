# load libraries
setwd('/home/barendregt/Analysis/PredictionError/data_output/')

num_timepoints = 110

# read data per time point & run ANOVA
pvals_correct <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=3,dimnames=list(c(), c("TR", "TI", "TR*TI"))))
pvals_incorrect <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=3,dimnames=list(c(), c("TR", "TI", "TR*TI"))))

fvals_correct <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=3,dimnames=list(c(), c("TR", "TI", "TR*TI"))))
fvals_incorrect <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=3,dimnames=list(c(), c("TR", "TI", "TR*TI"))))

fvals_all <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=2,dimnames=list(c(), c("type", "TR*TI*type"))))
pvals_all <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=2,dimnames=list(c(), c("type", "TR*TI*type"))))

for(t in 0:(num_timepoints-1)){
  timepoint_data <- read.csv(paste('pupil_data_t',t,'.csv', sep=''))
  
  rpm_all <- with(timepoint_data, aov(pupil ~ TR*TI*type + Error(Sub / (TR*TI*type))))
  
  fvals_all[t,1] <- unlist(summary(rpm_all))["Error: Sub:type.F value1"]
  fvals_all[t,2] <- unlist(summary(rpm_all))["Error: Sub:TR:TI:type.F value1"]
  pvals_all[t,1] <- unlist(summary(rpm_all))["Error: Sub:type.Pr(>F)1"]
  pvals_all[t,2] <- unlist(summary(rpm_all))["Error: Sub:TR:TI:type.Pr(>F)1"]  
}

# Plot results


# CORRECT TRIALS
plot_range = 21:100
fig_range = range(0, fvals_all$type[plot_range])#, fvals_incorrect$TR[plot_range])#, pvals_all$type[plot_range])


# p_value = (vector(length=num_timepoints)+1)*-log(0.05)
# plot(p_value[plot_range], type='l', col='grey', ylab='F', xlab='Time(s)', ylim=fig_range, axes=FALSE)

#axis(1, at=seq(0,num_timepoints,by=5), lab=seq(-1.5,4.5,by=.5))
# axis(2, at=2*0:fig_range[2], las=1)

plot(fvals_all$type[plot_range], type='l', col='blue', ylab='F-value', xlab ='Time(s)', ylim=fig_range, axes=FALSE)
points(seq(1,length(plot_range)+1)[pvals_all$type[plot_range]<0.05],fvals_all$type[plot_range][pvals_all$type[plot_range]<0.05],type='p',cex=2,pch='*',col='blue')

lines(fvals_all$'TR*TI*type'[plot_range], lty='solid',ylim=fig_range, col='black',lw=2)
points(seq(1,length(plot_range)+1)[pvals_all$'TR*TI*type'[plot_range]<0.05],fvals_all$'TR*TI*type'[plot_range][pvals_all$'TR*TI*type'[plot_range]<0.05],type='p',cex=2,pch='*',col='black')

axis(2, at=2*0:fig_range[2], las=1)
axis(1, at=seq(0,100,by=10), lab=seq(0.0,5.0,by=.5))

legend(0, fig_range[2], c("Correctness",'TR*TI*correct'), cex=0.8, col=c("blue",'black'),lty=c('solid','solid'),lw=c(1,2))
title('3-way test')

