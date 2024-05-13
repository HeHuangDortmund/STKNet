library(reticulate)
library(data.table)
library(ggplot2)
np <- import("numpy")

sum(res_DTW == 0)/length(res_DTW)


load_data = function(method, digit, n){
  predicted_label_ = paste0('predicted_label_test_for_digit_',digit,'_nTrain_',n,'.npy')
  true_label_ = paste0('true_label_test_for_digit_',digit,'_nTrain_',n,'.npy')
  predicted_label = np$load(file.path('results', method, 'BB0',predicted_label_))
  #true_label = np$load(file.path('results', method, 'BB0',true_label_))
  true_label = rep(digit, length(predicted_label))
  erg = cbind(method, n, predicted_label, true_label)
  erg = data.frame(erg)
  names(erg) = c('method','n','predicted_label', 'true_label')
  return(erg)
}


Methode = c('DTW', 'LSTM', 'Proposed')
digits = 0:9
ns = c(seq(10, 100, 10),120)

datas = data.frame(method = NULL,n = NULL,predicted_label = NULL, true_label = NULL)

for (methode in Methode) {
  for (digit in digits) {
    for (n in ns) {
      datas = rbind(datas, load_data(methode, digit, n))
    }
  }
}

datas = as.data.table(datas)
datas$correct = datas$predicted_label == datas$true_label
datas$method[datas$method == 'DTW'] = 'DTW+kNN'
datas$method[datas$method == 'LSTM'] = 'LSTM'
datas$method[datas$method == 'Proposed'] = 'STKNet+kNN'
datas$method = factor(datas$method)
total_acc = datas[, mean(correct), by = c("method", 'n')]
total_acc$n = as.numeric(total_acc$n)
total_acc$n = total_acc$n*10
str(total_acc)
total_acc$method
max(total_acc$V1)

LeNet5_acc = c(0.42149999737739563, 0.6538000106811523, 
           0.6626999974250793, 0.7264000177383423, 
           0.767300009727478, 0.7799000144004822,
           0.7955999970436096, 0.789900004863739, 
           0.8079000115394592, 0.8165000081062317,0.83)

LeNet5res = cbind('LeNet5', c(seq(100, 1000, 100), 1200),LeNet5_acc)
LeNet5res = as.data.table(LeNet5res)
names(LeNet5res) = c("method", "n", "V1")
LeNet5res$n = as.numeric(LeNet5res$n)
LeNet5res$V1 = as.numeric(LeNet5res$V1)
str(LeNet5res)
total_acc = rbind(total_acc, LeNet5res)
total_acc$data_type = 'Recoveried Trajectory'
total_acc$data_type[total_acc$method == 'LeNet5'] = 'Original Image'
total_acc$data_type = factor(total_acc$data_type, levels = c('Recoveried Trajectory from MNIST', 'Original MNIST Image'))
names(total_acc)[1] = 'Methods'
names(total_acc)[4] = 'Data'
total_acc$Methods = factor(total_acc$Methods, levels = c('STKNet+kNN',
                                                         'LSTM',
                                                         'DTW+kNN',
                                                         'LeNet5'))
ggplot(total_acc, aes(x = n, y = V1, col = Methods))+
  geom_line(aes(linetype = Data), size = 1)+
  geom_point(size = 3)+
  xlab('Number of training samples')+
  ylab('Accuracy on 10 000 test samples')+
  scale_x_continuous(breaks = c(seq(100, 1000, 100), 1200))
  

datasn100 = datas[n == 120]
bymethodLabel = datasn100[, .N, by = c('method', 'true_label', 'predicted_label')]
names(bymethodLabel)[2] = 'True Labels'
names(bymethodLabel)[3] = 'Predicted Labels'
library(RColorBrewer)
# 从ColorBrewer中选择一个调色板
my_palette <- brewer.pal(10, "Set3")

my_colors <- c("#E41A1C", "#377EB8", 
                            "#4DAF4A", "#984EA3",
                            "#FF7F00", "#FFFF33",
                            "#A65628", "#F781BF",
                            "#999999", "#000000")
ggplot(bymethodLabel, aes(x = `True Labels`, y = N , fill = `Predicted Labels`))+
  geom_bar(position="fill", stat="identity")+
  facet_wrap(~method)+
  scale_fill_manual(values = my_palette)+
  ylab('Proportion of each predicted label')+
  theme(text = element_text(size = 12))


