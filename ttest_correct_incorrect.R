# load libraries
setwd('/home/barendregt/Analysis/PredictionError/data_output/')

num_timepoints = 110

# read data per time point & run ANOVA
diff_data <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=4,dimnames=list(c(), c("pupil","p","s_low","s_up"))))


for(t in 1:num_timepoints){
  timepoint_data <- read.csv(paste('pupil_data_avg_t',(t-1),'.csv', sep=''))
  
  selection_c = timepoint_data$type=='correct' & timepoint_data$TR==1
  selection_in = timepoint_data$type=='incorrect' & timepoint_data$TR==1
  
  selection_pp_c = timepoint_data$type=='correct' & timepoint_data$TR==0 & timepoint_data$TI==0
  selection_pp_in = timepoint_data$type=='incorrect' & timepoint_data$TR==0 & timepoint_data$TI==0

 # diff_data[t,1] = mean(timepoint_data$pupil[selection_in]) - mean(timepoint_data$pupil[selection_c])
  # diff_data[t,2] = t.test(timepoint_data$pupil[selection_in],timepoint_data$pupil[selection_c],paired=FALSE)['p.value']

  #diff_data[t,2] = unlist(summary(with(timepoint_data,aov(pupil ~ type + Error(Sub / type)))))["Error: Sub:type.Pr(>F)1"]
  
  diff_data[t,1] = mean((timepoint_data$pupil[selection_in]-rep(timepoint_data$pupil[selection_pp_in],each=2)) - (timepoint_data$pupil[selection_c]-rep(timepoint_data$pupil[selection_pp_c])))
  #diff_data[t,3] = diff_data[t,1] - sd((timepoint_data$pupil[selection_in]-timepoint_data$pupil[selection_pp_in]) - (timepoint_data$pupil[selection_c]-timepoint_data$pupil[selection_pp_c]))/sqrt(33)
  #diff_data[t,4] = diff_data[t,1] + sd((timepoint_data$pupil[selection_in]-timepoint_data$pupil[selection_pp_in]) - (timepoint_data$pupil[selection_c]-timepoint_data$pupil[selection_pp_c]))/sqrt(33)
  diff_data[t,2] = t.test(timepoint_data$pupil[selection_in]-rep(timepoint_data$pupil[selection_pp_in],each=2),timepoint_data$pupil[selection_c]-rep(timepoint_data$pupil[selection_pp_c]),paired=TRUE)['p.value']

}

plot_range = 10:99

plot(diff_data$pupil[plot_range], type='l',ylab='Pupil size difference (incorrect-correct)',ylim=c(-0.1,0.30),axes=FALSE)
#segments(seq(1,length(plot_range)+1), diff_data$s_low[plot_range],seq(1,length(plot_range)+1), diff_data$s_up[plot_range])
points(seq(1,length(plot_range)+1)[diff_data$p[plot_range]<0.05],diff_data$pupil[plot_range][diff_data$p[plot_range]<0.05],pch=16,cex=2)

axis(1,seq(0,110,by=10),lab=seq(-0.5,5.0,by=.5))
axis(2,)