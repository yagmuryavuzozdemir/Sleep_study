#!/usr/bin/env Rscript
# 
# 
# require(reshape2)
# load("final_data.Rda")
# head(final_data)
# sleep = final_data[,-c(4,6:16,19:21,27,28,31:35)]
# head(sleep)
# #df = data.frame(sleep)
# df = as.data.frame(sleep)
# 
# # Grouping the data according to the subjects
# df_trans = dcast(df, name + Go_Bed + Age + Sex + `Annual Family Income` + RACE + BMI + WC + WCZ + WCHZ + `Maternal education` ~ TargetID, value.var="value")
# 
# # Scaling the numerical variables (TargetID)
# df_trans[,12:865929] = scale(df_trans[,12:865929])
# 
# df_trans$Sex = as.factor(df_trans$Sex)
# df_trans$RACE = as.factor(df_trans$RACE)
# df_trans$`Annual Family Income` = as.factor(df_trans$`Annual Family Income`) 
# df_trans$`Maternal education` = as.factor(df_trans$`Maternal education`)
# 
# # view some part of the data
# df_trans[, 1:20]
# 
# # Get the response and the features as y and X, respectively
# y <- as.factor(df_trans$Go_Bed)
# X <- df_trans[,-c(1, 2)]
# #X <- X[,1:1000]
# y_label <- ifelse(y== "E", 1, 0)  # transform the response to be 1s and 0s instead of characters
# 
# ### checking for missing values 
# has_missing_values <- any(is.na(X))
# 
# # Print the result
# if (has_missing_values) {
#   print("There are missing values in X.")
# } else {
#   print("There are no missing values in X.")
# }
# # no of missing values
# sum(is.na(X))
# 
# 
# #------------ Handling missing data --------------------------------------------#
# # function to find the columns with all NA values
# find_columns_with_all_NA <- function(dataframe) {
#   na_columns <- colnames(dataframe)[apply(dataframe, 2, function(x) all(is.na(x)))]
#   return(na_columns)
# }
# 
# # Function to remove columns with all NA values
# remove_columns_with_all_NA <- function(dataframe) {
#   na_columns <- apply(dataframe, 2, function(x) all(is.na(x)))
#   dataframe <- dataframe[, !na_columns, drop = FALSE]
#   return(dataframe)
# }
# 
# # print the columns with all NA vales
# na_columns <- find_columns_with_all_NA(X)
# print(na_columns)
# length(na_columns)
# 
# # Remove columns with all NA values
# X_new <- remove_columns_with_all_NA(X)
# 
# sum(is.na(X_new))
# 
# ### loading the packages
# 
# library(caret)
# 
# y_train = as.factor(y_label)
# x_train = X_new
# 
# 
# save(y_train, x_train, file = "data_last.Rda")

load("data_last.Rda")
# 
# index_screen = seq(1,ncol(x_train))
# crit = rep(NA, ncol(x_train))
# 
# # data storage
# Criteria <- list()
# Group <- list()
# selected_group <- list()

### CV parameters
# 
# library(caret)
# 
# regressControl  <- trainControl(method="none")
# 
# ### Screening step of SWAG
# 
# # for(i in seq_along(index_screen)){
# #   # index of group of variables
# #   
# #   X = x_train[,index_screen[[i]]]
# #   fit_glm_cve = try(train(y = y_train, x = data.frame(X),
# #                           method  = "glm", family = "binomial",
# #                           trControl = regressControl),silent = TRUE)
# #   crit[i] = AIC(fit_glm_cve$finalModel)
# #   
# # }
# # 
# # save(crit, file = "screening_AIC.Rda")

#load("screening_AIC.Rda")

# data storage
# Criteria <- list()
# Group <- list()
# selected_group <- list()
# 
# ### CV parameters
# 
# regressControl  <- trainControl(method="none")

### General step of SWAG

# parameters

q0 = 0.001
dmax = 4
# graine = 11
# Criteria[[1]] <- crit 
# Group[[1]] <- seq_along(crit) 
# selected_group[[1]] <- which(crit <= quantile(crit,q0)) #models with smaller error
# id_screening <- selected_group[[1]]

# adding the targetid's that is expected to be found

# genes = read.csv("genes_to_be_added.csv")
# 
# add_ind = which(colnames(x_train) %in% genes[, 1])
# 
# id_screening = union(id_screening,add_ind)
# 
# id_screning = scale(id_screening)
# 
# m = choose(length(id_screening),2)

#### Dealing with missing values 

#library(VIM)

# Perform KNN imputation

#data_screened = kNN(x_train[,id_screening], k = 6, imp_var = F)

#for(i in 1:length(id_screening)){
 # x_train[,id_screening[i]] = data_screened[,i]
#}

###### SWAG GENERAL STEP #######


# for(d in 2:dmax){
#   
#   # cv0 <- cv1
#   idRow <- selected_group[[d-1]] 
#   
#   if(d==2){ 
#     idVar <- Group[[d-1]][idRow] 
#     nrv <- length(idVar)
#   }else{ 
#     if(length(idRow) == 1){
#       idVar <- Group[[d-1]][idRow]
#       nrv <- length(idVar)
#     }else{
#       idVar <- Group[[d-1]][idRow,]
#       nrv <- nrow(idVar)
#     }
#   }
#   # build all possible 
#   A <- matrix(nr=nrv*length(id_screening),nc=d) 
#   A[,1:(d-1)] <- kronecker(cbind(rep(1,length(id_screening))),idVar) 
#   A[,d] <- rep(id_screening,each=nrv) 
#   B <- unique(t(apply(A,1,sort))) 
#   id_ndup <- which(apply(B,1,anyDuplicated) == 0) 
#   var_mat <- B[id_ndup,] 
#   
#   rm(list=c("A","B"))
#   
#   if(nrow(var_mat)>m){
#     set.seed(graine+d) 
#     Group[[d]] <- var_mat[sample.int(nrow(var_mat),m),]
#   }else{
#     Group[[d]] <- var_mat
#   }
#   
#   var_mat <- Group[[d]]
#   
#   crit = rep(NA,nrow(var_mat))
#   
#   for(i in seq_along(crit)){
#     # index of group of variables
#     
#     X = x_train[,var_mat[i,]]
#     
#     fit = train(y = y_train, x = data.frame(X),
#                             method  = "glm", family = "binomial",
#                             trControl = regressControl)
#     crit[i] = AIC(fit$finalModel)
#     print(c(d,i))
#   }
#   
#   
#   index.na = !is.na(crit)
#   Criteria[[d]] <- crit[index.na] 
#   Group[[d]] = var_mat[index.na,]
#   cv1 <- quantile(crit,probs=q0,na.rm=T)
#   selected_group[[d]] <- which(crit<=cv1)
#   
# }
# 
# 
# save(Criteria,Group, selected_group, id_screening, file = "sleep_aic.Rda")

##### ANALYSIS OF THE RESULTS #####

load("sleep_aic.Rda")

### Necessary libraries

library(plyr)
library(igraph)

############ POST PROCESSING ########

crit_post = list() #index set of the models
post_group = list() #models 

# storing all AIC's 
all_aic = c()

for(i in 2:dmax){
  all_aic = c(all_aic,Criteria[[i]])
}

aic = quantile(all_aic, probs=q0, na.rm = T)

for(i in 2:dmax){
  crit_post[[i]] = which(Criteria[[i]]<=aic)
  post_group[[i]] = Group[[i]][crit_post[[i]],]
  
}

x = c() #non-empty elements of post-group
for(i in 1:dmax){
  if(length(post_group[[i]])!=0){
    x = c(x,i)
  }
  x = x
}

#completing all elements of post-group into a matrix of col.size 6
for(i in 1:length(x)){
  if(ncol(post_group[[x[i]]])<6){
    diff = dmax - ncol(post_group[[x[i]]])
    post_group[[x[i]]] = cbind(post_group[[x[i]]],matrix(NA,ncol = diff, nrow = nrow(post_group[[x[i]]])))
  } 
}

#combining the models into one matrix
post_models = matrix(NA, ncol = dmax,nrow=nrow(post_group[[x[1]]]))

post_models[1:nrow(post_group[[x[1]]]),]=post_group[[x[1]]]

for(i in 2:length(x)){
  post_models = rbind(post_models,post_group[[x[i]]])
}

# Total number of models after post processing is 4210 dimensions varying from 2 to 4. 

# frequency table of the post models

selected_var_post = c() 
for(i in 1:ncol(post_models)){
  selected_var_post = c(selected_var_post,post_models[,i])
}
selected_var_post = na.omit(unique(selected_var_post))
selected_var_post = sort(selected_var_post)

freq_post = table(post_models)
variable = colnames(x_train)[selected_var_post]

freq_table_post = cbind(variable,freq_post)
rownames(freq_table_post) = c(1:nrow(freq_table_post))
freq_table_post = as.data.frame(freq_table_post)
freq_table_post$freq_post = as.numeric(freq_table_post$freq_post)
freq_table_post$percentage = round(freq_table_post$freq_post/nrow(post_models),3)
# Order the data frame by the 'percentage' column in decreasing order
freq_table_post <- freq_table_post[order(-freq_table_post$percentage), ]

### Checking the related genes with selected ones

genes = read.csv("genes_to_be_added.csv")

genes$Name = as.character(genes$Name)

#common = intersect(genes$Name, freq_table_post$variable)

#which(freq_table_post$variable %in% common , arr.ind = T)

### Coefficient of the variables in each model

betas = matrix(NA, ncol=3, nrow=nrow(post_models))
for(i in 1:nrow(post_models)){
  ind_models = post_models[i,]
  ind_models = unlist(na.omit(ind_models))
  data = as.data.frame(cbind(y_train, x_train[,ind_models]))
  names(data) = c("y_train", names(x_train[,ind_models]))
  betas[i,1:length(ind_models)]=glm(as.factor(y_train)~ ., data = data, family = "binomial")$coefficients[-1]
}


sign_betas = list()
for(i in 1:length(selected_var_post)){
  sign_betas[[i]] = betas[which(post_models==selected_var_post[i],arr.ind = T)]
}

which(variable==freq_table_post[1,1])
which(variable==freq_table_post[2,1])
which(variable==freq_table_post[3,1])
which(variable==freq_table_post[4,1])
which(variable==freq_table_post[5,1])
which(variable==freq_table_post[6,1])
which(variable==freq_table_post[8,1])
which(variable==freq_table_post[10,1])
which(variable==freq_table_post[11,1])
which(variable==freq_table_post[12,1])








which(variable==freq_table_post[1,1])### Intensity matrix (shows the number of times two variables are in the same model)

which(variable==freq_table_post[1,1])A = matrix(0, nrow = ncol(post_models), ncol =ncol(post_models))
which(variable==freq_table_post[1,1])intensity = matrix(0, nrow = length(selected_var_post), ncol = length(selected_var_post))
b=0
a = list()
for(i in 1:(length(selected_var_post)-1)){
  for(j in (i+1):length(selected_var_post)){
    for(k in 1:(ncol(post_models)-1)){
      a[[i]]=which(post_models[,k]==selected_var_post[i])
      for(n in (k+1):(ncol(post_models))){
        A[k,n]=length(which(post_models[a[[i]],n]==selected_var_post[j]))
      }
    }
    intensity[i,j]=sum(A)
    intensity[j,i]=sum(A)
  }
}

colnames(intensity) = selected_var_post

rownames(intensity) = selected_var_post

#variables that are mostly together
which(intensity==max(intensity),arr.ind = T)
colnames(x_train)[] #row of the max
colnames(x_train)[] #column of the max

### Creating the SWAG network

g = graph_from_adjacency_matrix(intensity, mode = "undirected")  


a = read.csv("manifest.csv")
b = read.csv("sleep.csv")
c = read.csv("frequency.csv")

overlap = intersect(b$TARGET.IDS.ASS..W.SLEEP.RELATED.GENES, c$variable)

ord = c()
for(i in 1:length(overlap)){
  ord = c(ord,which(c$variable==overlap[i]))
}


ord = sort(ord)
save(ord, file = "order.Rda")

id_gene = c$variable[ord]

# ind = intersect(a$UCSC_RefGene_Name,b$sleep.related.genes)
# index = c()
# for(i in 1:length(ind)){
#   index = c(index, which(a$UCSC_RefGene_Name == ind[i]))
# }
# 
# sleep_targetid = a$Name[index]

### Extracting the 4 most important variables for MANOVA 

variables = selected_var_post
freq_index = cbind(variables,freq_post)

rownames(freq_index) = c(1:nrow(freq_index))
freq_index = as.data.frame(freq_index)
freq_index$freq_post = as.numeric(freq_index$freq_post)
freq_index$percentage = round(freq_index$freq_post/nrow(post_models),3)

imp = freq_index[order(freq_index$freq_post,decreasing = T),][c(1,2,3,4,5,6,8,10,11,13),1]

x_imp = as.matrix(x_train[imp])
#model = manova(x_imp~y_train)
#summary(model)
#summary.aov(model) # all features are significant one by one, which is natural since they are screened (1 dimensional model)

library(ggplot2)
library(gridExtra)

sign_betas_imp = sign_betas[c(375,837,35,935,362,984,306,180,136,7)]

sign_betas_imp = lapply(sign_betas_imp, na.omit)



# Create a sample list of 10 vectors, each with random values
data <- data.frame(
  group = factor(c("cg09760986", "cg22792063", "cg00807892", "cg25282780", 
                   "cg09321097", "cg26811976", "cg07891983", 
                   "cg04402799", "cg03232960", "cg00124101")),
  Q1_value = sapply(sign_betas_imp, function(x) quantile(x, 0.25)),
  median_value = sapply(sign_betas_imp, function(x) quantile(x, 0.50)),
  Q3_value = sapply(sign_betas_imp, function(x) quantile(x, 0.75)),
  percentage_range = sample(1:6, 10, replace = TRUE)
)


# Define color palette for percentage ranges
colors <- c("#053061", "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7")

# Plot
ggplot(data) +
  # Horizontal line segments for range (Q1 to Q3), in blue
  geom_errorbarh(aes(y = group, xmin = Q1_value, xmax = Q3_value, color = "IQR for Associative β"), 
                 height = 0.2, size = 1) +
  # Median points in orange
  geom_point(aes(x = median_value, y = group, color = "Associative β median"), size = 2) +
  # Single values in purple
  #geom_point(aes(x = single_value, y = group, color = "Single β"), size = 3) +
  # Dashed vertical line at x = 0
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  # Custom legend
  scale_color_manual(name = NULL,
                     values = c("IQR for Associative β" = "blue", 
                                "Associative β median" = "orange", 
                                "Single β" = "purple"),
                     labels = c( "Associative β median","IQR for Associative β", "Single β")) +
  # Labels and theme adjustments
  labs(x = "Value", y = NULL) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 12, color = "black"), # Adjust y-axis text style
    legend.position = "right", # Position legend to the right
    legend.title = element_text(size = 10), # Legend title size
    legend.text = element_text(size = 8) # Legend text size
  ) +
  # Adjust legend appearance
  guides(color = guide_legend(override.aes = list(
    list(linetype = "solid", color = "blue", size = 1),   # For "Range for Associative β" as a blue line
    list(shape = 16, size = 4, color = "orange"),         # For "Associative β median" as an orange dot
    list(shape = 16, size = 3, color = "purple")          # For "Single β" as a purple dot
  )))




data_10 = data.frame(y_train, x_imp)

colnames(data_10) = c("gotobed", colnames(x_imp))

data_10$gotobed = ifelse(data_10$gotobed == 1, "Early", "Late")

save(data_10, x_imp, file = "boxplot_data.Rda")

p_values <-rep(0.001, 10)

# Define a function to create the boxplots with titles and p-values
create_boxplot <- function(index, label, title, p_value) {
  ggplot(data_10, aes(x = factor(1), y = x_imp[, index], fill = gotobed)) +
    geom_boxplot() + 
    ylab(label) + 
    xlab(NULL) +  # Remove x-axis label
    ggtitle(title) +  # Add the title on top
    guides(fill = guide_legend(title = " ")) +
    theme_gray() +  # Use the default gray background
    theme(
      axis.text.x = element_blank(), 
      axis.ticks.x = element_blank(),
      legend.position = "none",    # Remove legend for each plot
      plot.title = element_text(hjust = 0.5),  # Center the title
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_blank(),# Remove vertical gridlines
      panel.grid.minor = element_blank()
    ) +
    # Add p-values centered at the top
    annotate(
      "text", x = 1, y = max(x_imp[, index], na.rm = TRUE) * 1.1, 
      label = paste("p <", p_value), 
      size = 4, color = "black", hjust = 0.5
    )
}

# List of labels and related titles
labels <- rep("methylation %", 10)

titles <- c("ABCG2", "ABHD4", "MOBKL1A", "AK3", 
            "SDE2", "PRAMEF4", "CREM", "CDH4", 
            "BRAT1", "SDK1")  # Add related names here

# Create the plots
plots <- lapply(1:10, function(i) create_boxplot(i, labels[i], titles[i], p_values[i]))

# Arrange the plots in a grid with 5 columns and 2 rows
grid.arrange(grobs = plots, ncol = 5, nrow = 2)










# Install cowplot if needed
# install.packages("cowplot")
library(cowplot)

# Rename 'labels' to 'plot_labels' to avoid conflicts
plot_labels <-c("cg09760986", "cg22792063", "cg00807892", "cg25282780", 
                "cg09321097", "cg26811976", "cg07891983", 
                "cg04402799", "cg03232960", "cg00136968")

titles <-c("ABCG2", "ABHD4", "MOBKL1A", "AK3", 
           "C1orf55", "PRAMEF4", "CREM", "CDH4", 
           "C7orf27", "SDK1")  # Related names

# Create a function that keeps the legend
create_boxplot_with_legend <- function(index, label, title) {
  ggplot(data_10, aes(y = x_imp[, index], fill = gotobed)) +
    geom_boxplot() + 
    ylab(label) + 
    xlab(NULL) +  # Remove the x-axis label
    ggtitle(title) +  # Add the title on top
    scale_fill_manual(values = c("lightblue", "lightcoral"), labels = c(expression(bold("Early")),expression(bold("Late"))), name = NULL) +  guides(fill = guide_legend(direction = "horizontal")) +# Set labels to Early and Late, remove title
    theme(
      legend.position = "right",           # Legend to be extracted
      plot.title = element_text(hjust = 0.5),  # Center the title
      axis.text.x = element_text(face = "bold")  # Make x-axis labels bold
    )
}

# Extract the legend from a sample plot
sample_plot <- create_boxplot_with_legend(1, plot_labels[1], titles[1])
legend <- get_legend(sample_plot)

# Create the plots without the legend
create_boxplot <- function(index, label, title) {
  ggplot(data_10, aes(y = x_imp[, index], fill = gotobed)) +
    geom_boxplot(show.legend = FALSE) +  # Remove the legend from the plot
    ylab(label) + 
    xlab(NULL) +  # Remove the x-axis label
    ggtitle(title) +  # Add the title on top
    scale_fill_manual(values = c("lightblue", "lightcoral"),  labels = c(expression(bold("Early")),expression(bold("Late"))), name = NULL) +  # Set labels to Early and Late, remove title
    theme(
      legend.position = "none",            # Completely remove the legend
      plot.title = element_text(hjust = 0.5),  # Center the title
      axis.text.x = element_text(face = "bold")  # Make x-axis labels bold
    )
}

# Create the plots with titles
plots <- lapply(1:10, function(i) create_boxplot(i, plot_labels[i], titles[i]))

# Arrange the plots in a grid with the legend
final_plot <- plot_grid(
  plot_grid(plotlist = plots, ncol = 5, nrow = 2),
  legend,
  ncol = 1,
  rel_heights = c(1, 0.1)  # Adjust height for the legend
)

# Display the final plot
print(final_plot)











create_boxplot <- function(index, label, title) {
  ggplot(data_10, aes(y = x_imp[, index], fill = gotobed)) +
    geom_boxplot(show.legend = FALSE) + # Remove the legend from the plot
    ylab(label) + 
    xlab(NULL) +  # Remove the x-axis label
    ggtitle(title) +  # Add the title on top
    scale_fill_manual(values = c("lightblue", "lightcoral"), labels = c(expression(bold("Early")),expression(bold("Late")))) +  # Set labels to Early and Late
    guides(fill = guide_legend(title = " ")) +
    theme(
      legend.position = "none",             # Place legend on the right
      plot.title = element_text(hjust = 0.5),  # Center the title
      axis.text.x = element_text(face = "bold")  # Make x-axis labels bold
    )
}

# Create the plots with titles
plots <- lapply(1:10, function(i) create_boxplot(i, labels[i], titles[i]))

# Arrange the plots in a grid
grid.arrange(grobs = plots, ncol = 5, nrow =2)



