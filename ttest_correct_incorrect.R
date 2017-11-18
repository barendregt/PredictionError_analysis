# load libraries
setwd('/home/barendregt/Analysis/PredictionError/data_output/')

num_timepoints = 110

# read data per time point & run ANOVA
diff_data <- as.data.frame(matrix(0,nrow=num_timepoints,ncol=2,dimnames=list(c(), c("pupil","p"))))


for(t in 0:(num_timepoints-1)){
  timepoint_data <- read.csv(paste('pupil_data_t',t,'.csv', sep=''))
  
  selection_c = timepoint_data$type=='correct' & timepoint_data$TR==1
  selection_in = timepoint_data$type=='incorrect' & timepoint_data$TR==1
  
  selection_pp_c = timepoint_data$type=='correct' & timepoint_data$TR==0 & timepoint_data$TI==0
  selection_pp_in = timepoint_data$type=='incorrect' & timepoint_data$TR==0 & timepoint_data$TI==0
  
  # diff_data[t,1] = mean((timepoint_data$pupil[selection_in]) - (timepoint_data$pupil[selection_c]))
  # diff_data[t,2] = t.test(timepoint_data$pupil[selection_in],timepoint_data$pupil[selection_c],paired=TRUE)['p.value']
  # 
  diff_data[t,1] = mean((timepoint_data$pupil[selection_in]-timepoint_data$pupil[selection_pp_in]) - (timepoint_data$pupil[selection_c]-timepoint_data$pupil[selection_pp_c]))
  diff_data[t,2] = t.test(timepoint_data$pupil[selection_in]-timepoint_data$pupil[selection_pp_in],timepoint_data$pupil[selection_c]-timepoint_data$pupil[selection_pp_c],paired=TRUE)['p.value']

}

plot_range = 21:100

plot(diff_data$pupil[plot_range], type='l',ylab='Pupil size difference (incorrect-correct)', axes=FALSE)
points(seq(1,length(plot_range)+1)[diff_data$p[plot_range]<0.05],diff_data$pupil[plot_range][diff_data$p[plot_range]<0.05],pch='*',cex=2)

axis(1,seq(0,110,by=10),lab=seq(0,5.5,by=.5))
axis(2,)