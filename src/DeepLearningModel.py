

import os
import sys
sys.path.insert(1,'src/graph-lrp/lib')
import models
import graph
import coarsening
sys.path.insert(2,'src/graph-lrp/components')
import data_handling
import glrp_scipy
os.chdir('src/graph-lrp/')
print('Success!!')
import yaml
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from csv import writer
file_path_template = 'src/genes_main_config.yml'

#if not os.path.isdir(file_path_template):
#    print('Enter the right path')
#    sys.exit()
file = open(file_path_template,'r')
cfc = yaml.load(file,Loader=yaml.FullLoader)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


"""
Running the GLRP on GCNN model trained on gene expression data. 90% is for training and 10% for testing. 
Relevances obtained for 10% of testing patients are written into the file "relevances_rendered_class.csv". From these relevances the patient subnetworks can be built.
The file "predicted_concordance.csv" contains a table showing which patients were predicted correctly.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# from sklearn.model_selection import StratifiedKFold

import time

rndm_state = 7
np.random.seed(rndm_state)


path_to_feature_val = str(cfc['input_files']['path_to_feature_val'])
path_to_feature_graph = str(cfc['input_files']['path_to_feature_graph'])
path_to_labels = str(cfc['input_files']['path_to_labels'])
DP = data_handling.DataPreprocessor(path_to_feature_values=path_to_feature_val, path_to_feature_graph=path_to_feature_graph,path_to_labels=path_to_labels)
X = DP.get_feature_values_as_np_array()  # gene expression
A = csr_matrix(DP.get_adj_feature_graph_as_np_array().astype(np.float32))  # adjacency matrix of the PPI network
y = DP.get_labels_as_np_array()  # labels
print("GE data, X shape: ", X.shape)
print("Labels, y shape: ", y.shape)
print("PPI network adjacency matrix, A shape: ", A.shape)
X_train_unnorm, X_test_unnorm, y_train, y_test = train_test_split(X, y, test_size=cfc['dl_params']['test_ratio'],stratify=y, random_state=rndm_state)
_, _, patient_indexes_train, patient_indexes_test = train_test_split(X, DP.labels.columns.values.tolist(), test_size=cfc['dl_params']['test_ratio'],stratify=y, random_state=rndm_state)
patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})
X_train = X_train_unnorm - np.min(X)
X_test = X_test_unnorm - np.min(X)
print("X_train max", np.max(X_train))
print("X_train min", np.min(X_train))
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train, shape: ", y_train.shape)
print("y_test, shape: ", y_test.shape)
graphs, perm = coarsening.coarsen(A, levels=2, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]
X_train = coarsening.perm_data(X_train, perm)
X_test = coarsening.perm_data(X_test, perm)
n_train = X_train.shape[0]
params = dict()
params['dir_name']       = 'GE'
params['num_epochs']     = int(cfc['dl_params']['epochs'])
params['batch_size']     = int(cfc['dl_params']['batch_size'])
params['eval_frequency'] = cfc['dl_params']['eval_freq']
params['filter']         = str(cfc['dl_params']['filter'])
params['brelu']          = str(cfc['dl_params']['brelu'])
params['pool']           = str(cfc['dl_params']['pool'])
C = y.max() + 1
assert C == np.unique(y).size
params['F']              = [int(cfc['dl_params']['graph_cnn_filters']), int(cfc['dl_params']['graph_cnn_filters'])]  # Number of graph convolutional filters.
params['K']              = [int(cfc['dl_params']['polynomial_ord']),int(cfc['dl_params']['polynomial_ord'])]  # Polynomial orders.
params['p']              = [int(cfc['dl_params']['pooling_size']),int(cfc['dl_params']['pooling_size']) ]    # Pooling sizes.
params['M']              = [512, 128, C]  # Output dimensionality of fully connected layers.
params['regularization'] = cfc['dl_params']['regularization']
params['dropout']        = cfc['dl_params']['dropout']
params['learning_rate']  = cfc['dl_params']['learning_rate']
params['decay_rate']     = cfc['dl_params']['decay_rate']
params['momentum']       = cfc['dl_params']['momentum']
params['decay_steps']    = n_train / params['batch_size']
model = models.cgcnn(L, **params)

    #!!!
    #Running ML methods
#clf_ob = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#y_pred = clf_ob.predict(X_test)
#
#rf = RandomForestClassifier(random_state=1,n_estimators=12, min_samples_leaf=1)
#rf.fit(X_train,y_train)
#rf_pred = rf.predict(X_test)
#
#logReg_l1 = LogisticRegression(l1_ratio=1,penalty = 'l1', random_state=1, solver='saga').fit(X_train, y_train)
#logReg_elastic = LogisticRegression(l1_ratio=0.5,penalty = 'elasticnet', random_state=1, solver='saga').fit(X_train, y_train)
#logReg_l1_pred = logReg_l1.predict(X_test)
#logReg_elastic_pred = logReg_elastic.predict(X_test)
#
#print("Accuracy SVM:",metrics.accuracy_score(y_test, y_pred))
#print("Accuracy RandomForest:",metrics.accuracy_score(y_test, rf_pred))
#print("Accuracy LogReg L1 penalty:",metrics.accuracy_score(y_test, logReg_l1_pred))
#print("Accuracy LogReg elasticnet penalty:",metrics.accuracy_score(y_test, logReg_elastic_pred))

    # !!!
    # TRAINING.
config_i = file_path_template.split('/')
config_i = config_i[len(config_i)-1]
out_file = '/home/ppugale/OutStats/' + 'Statistics_For_Reg_'+  config_i + '.pdf'
print(out_file)
start = time.time()
accuracy, loss, t_step, trained_losses, learning_rates, epoch_rec = model.fit(X_train, y_train, X_test, y_test)
print("Testing the print statement for accuracy", accuracy)
end = time.time()
probas_ = model.get_probabilities(X_test)
labels_by_network = np.argmax(probas_, axis=1)
fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
f1 = 100 * f1_score(y_test, labels_by_network, average='weighted')
acc = accuracy_score(y_test, labels_by_network)
print("\n\tTest AUC:", roc_auc) # np.argmax(y_preds, axis=2)[:, 0] fot categorical
print("\tTest F1 weighted: ", f1)
print("\tTest Accuraccy:", acc, "\n")



with PdfPages(out_file) as pdf:
    fig1 = plt.figure(figsize=(11.69,8.27))
    df_to_plot = pd.DataFrame({'Epochs':epoch_rec, 'LearningRate': learning_rates, 'TrainedLoss': trained_losses, 'Loss': loss, 'Accuracy': accuracy})
    plt.subplot(221)
    plt.plot( 'Epochs', 'LearningRate', data=df_to_plot, marker='o', alpha=0.4)
    plt.title('Learning Rate', fontsize=8, color='grey', loc='left', style='italic')
    plt.subplot(222)
    plt.plot( 'Epochs','TrainedLoss', data=df_to_plot, linestyle='none', marker='o', color="orange", alpha=0.3)
    plt.title('Trained Loss', fontsize=8, color='grey', loc='left', style='italic')
    plt.subplot(223)
    plt.plot( 'Epochs','Loss', data=df_to_plot, linestyle='none', marker='D', color="green", alpha=0.3)
    plt.title('Loss ', fontsize=8, color='grey', loc='left', style='italic')
    plt.subplot(224)
    plt.plot( 'Epochs','Accuracy', data=df_to_plot, marker='o', color="grey", alpha=0.3)
    plt.title('Accuracy', fontsize=8, color='grey', loc='left', style='italic')
    title_plt = 'Statistics_For_Reg_'+ str(params['regularization']) + '_LR_' + str(params['learning_rate']) + '_DR_' + str          (params['decay_rate'])
    plt.suptitle(title_plt)
    pdf.savefig(fig1)
    plt.close()
    fig2 = plt.figure(figsize=(11.69,8.27))
    fig2.clf()
    txt = "Test AUC:"+ str(roc_auc)+ '\n' + "Test F1 weighted: " +  str(f1)+ '\n' + "Test Accuraccy:" +  str(acc) 
    fig2.text(0.5,0.5,txt, transform=fig2.transFigure, size=12, ha="center")
    pdf.savefig()
    plt.close()
    fig3 = plt.figure(figsize=(11.69,8.27))
    fig3.clf()
    txt = 'Paramters: ' + '\n' + 'Test Ratio' + str(cfc['dl_params']['test_ratio'])
    for k in params.keys():
      txt = txt + '\n' +str(k) + ': ' + str(params[k])
    fig3.text(0.5,0.5,txt, transform=fig3.transFigure, size=12, ha="center")
    pdf.savefig()
    plt.close()
    fig4 = plt.figure(figsize=(11.69,8.27))
    fig4.clf()
    txt = "Files Used:" + "\n" + "X: " + path_to_feature_val + "\n" + "Y: "+path_to_labels  + "\n" + "PPI: " +path_to_feature_graph
    fig4.text(0.5,0.5,txt, transform=fig4.transFigure, size=12, ha="center")
    pdf.savefig()
    plt.close()


list_data = [roc_auc,f1,acc,path_to_feature_val,cfc['dl_params']['test_ratio']]
for k in params.keys():
  list_data.append(params[k])
with open('/home/ppugale/Results_GLRP_Parameters_Accuracy.csv', 'a', newline='') as f_object:  
    writer_object = writer(f_object)
    writer_object.writerow(list_data)  
    f_object.close()

    # !!!
    # Creating hot-encoded labels for GLRP

I = np.eye(C)
tmp = I[labels_by_network]
labels_hot_encoded = np.ones((model.batch_size, C))
labels_hot_encoded[0:tmp.shape[0], 0:tmp.shape[1]] = tmp
print("labels_hot_encoded.shape", labels_hot_encoded.shape)
dir_to_save = str(cfc['output_loc']['res_dir'])
print("labels_by_network type", labels_by_network.dtype)
print("y_test type", y_test.dtype)
concordance = y_test == labels_by_network
concordance = concordance.astype(int)
out_labels_conc_df = pd.DataFrame(np.array([labels_by_network, concordance]).transpose(),columns=["Predicted", "Concordance"])
concordance_df = patient_ind_test_df.join(out_labels_conc_df)
concordance_df.to_csv(path_or_buf = dir_to_save + "predicted_concordance.csv", index=False)

    # !!!
    # CALCULATION OF RELEVANCES
    # CAN TAKE QUITE SOME TIME (UP to 10 MIN, Intel(R) Xeon(R) CPU E5-1620 v2 @ 3.70GHz, 32 GB RAM)
    
    
glrp = glrp_scipy.GraphLayerwiseRelevancePropagation(model, X_test, labels_hot_encoded)
rel = glrp.get_relevances()[-1][:X_test.shape[0], :]
rel = coarsening.perm_data_back(rel, perm, X.shape[1])
rel_df = pd.DataFrame(rel, columns=DP.feature_names)
rel_df = pd.DataFrame(data={"Patient ID": patient_indexes_test}).join(rel_df)
rel_df.to_csv(path_or_buf = dir_to_save + "relevances_rendered_class.csv", index=False)

