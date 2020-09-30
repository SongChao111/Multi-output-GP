import numpy as np
np.random.seed(206)

import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import theano
import theano.tensor as tt
import pymc3 as pm
import scipy.stats as st
# import arviz as az
import random

# This function is used to calculate the cwc index by pinrw and picp as input
def cwc_cal(pinrw, picp, mu=0.8, eta=50):
  gamma = 0 if picp >= mu else 1
  cwc = pinrw*(1 + gamma*np.exp(-eta * (picp-mu)))
  return cwc

# This function is used to calculate the index PICP and pinrw and CWC and so on
# The input is prediction of model and true measured value
def index_cal(y_pred, y_true, conf_int=0.95):
  # conf_int = 0.95
  alpha = 1-conf_int
  n = np.shape(y_true)[0]
  n_samples = np.shape(y_pred)[0]
  y_pred_mu = np.mean(y_pred,axis=0)
  y_pred_sd = np.std(y_pred,axis=0)

  # Calculate the lower bound and upper bound of 95% confidence interval
  y_pred_L = y_pred_mu - scipy.stats.norm.ppf(1-alpha/2) * y_pred_sd
  y_pred_U = y_pred_mu + scipy.stats.norm.ppf(1-alpha/2) * y_pred_sd

  coverage = np.zeros(n)
  for i in range(n):
    if (y_true[i] > y_pred_L[i]) & (y_true[i] < y_pred_U[i]):
      coverage[i] = 1
    else:
      coverage[i] = 0
  # prediction interval coverage probability
  picp = np.sum(coverage) / n
  R = np.max(y_true) - np.min(y_true)
  # mean prediction interval width
  mpiw = np.sum(y_pred_U-y_pred_L) / n
  # normalized mean prediction interval width
  nmpiw = mpiw / R
  # root-mean-square prediction interval width
  rpiw = (y_pred_U-y_pred_L)*(y_pred_U-y_pred_L)
  rpiw = np.sqrt(np.sum(rpiw)/n)
  # normalized root-mean-square prediction interval width
  pinrw = rpiw / R
  # CWC
  cwc = cwc_cal(pinrw, picp, mu=0.8)
  return pd.DataFrame([picp, mpiw, nmpiw, rpiw, pinrw, cwc],index=['picp', 'mpiw', 'nmpiw', 'rpiw', 'pinrw', 'cwc'])

# This is the function used for bayesian calibration
def Bayesian_Calibration(DataComp,DataField,DataPred,output_folder):
  # This is data preprocessing part
  n = np.shape(DataField)[0] # number of measured data
  m = np.shape(DataComp)[0] # number of simulation data

  p = np.shape(DataField)[1] -1 # number of input x
  q = np.shape(DataComp)[1] - p -1 # number of calibration parameters t

  xc = DataComp[:,1:] # simulation input x + calibration parameters t
  xf = DataField[:,1:] # observed input

  yc = DataComp[:,0] # simulation output
  yf = DataField[:,0] # observed output

  x_pred = DataPred[:,1:] # design points for predictions
  y_true = DataPred[:,0] # true measured value for design points for predictions
  n_pred = np.shape(x_pred)[0] # number of predictions
  N = n+m+n_pred

  # Put points xc, xf, and x_pred on [0,1] 
  for i in range(p):
    x_min = min(min(xc[:,i]),min(xf[:,i]))
    x_max = max(max(xc[:,i]),max(xf[:,i]))
    xc[:,i] = (xc[:,i]-x_min)/(x_max-x_min)
    xf[:,i] = (xf[:,i]-x_min)/(x_max-x_min)
    x_pred[:,i] = (x_pred[:,i]-x_min)/(x_max-x_min)

  # Put calibration parameters t on domain [0,1]
  for i in range(p,(p+q)):
    t_min = min(xc[:,i])
    t_max = max(xc[:,i])
    xc[:,i] = (xc[:,i]-t_min)/(t_max-t_min)

  # standardization of output yf and yc
  yc_mean = np.mean(yc)
  yc_sd = np.std(yc)
  yc = (yc-yc_mean)/yc_sd
  yf = (yf-yc_mean)/yc_sd

  # This is modeling part
  with pm.Model() as model:
    # Claim prior part
    eta = pm.HalfCauchy("eta", beta=3) # for eta of gaussian process
    lengthscale = pm.Gamma("lengthscale", alpha=2, beta=1, shape=(p+q)) # 2,1 for lengthscale of gaussian process
    tf = pm.Beta("tf", alpha=2, beta=2, shape=q) # for calibration parameters
    sigma1 = pm.HalfCauchy('sigma1', beta=5) # for noise
    y_pred = pm.Normal('y_pred', 0, 1.5, shape=n_pred) # for y prediction

    # Concate data into a big matrix[[xf tf], [xc tc], [x_pred tf]]
    xf1 = tt.concatenate([xf, tt.fill(tt.zeros([n,q]), tf)], axis = 1)
    x_pred1 = tt.concatenate([x_pred, tt.fill(tt.zeros([n_pred,q]), tf)], axis = 1)
    X = tt.concatenate([xf1, xc, x_pred1], axis = 0)
    # Concate data into a big matrix[[yf], [yc], [y_pred]]
    y = tt.concatenate([yf, yc, y_pred], axis = 0)

    # Covariance funciton of gaussian process
    cov_z = eta**2 * pm.gp.cov.ExpQuad((p+q), ls=lengthscale)
    # Gaussian process with covariance funciton of cov_z
    gp = pm.gp.Marginal(cov_func = cov_z)
    # Bayesian inference
    outcome = gp.marginal_likelihood("outcome", X=X, y=y, noise=sigma1)
    trace = pm.sample(250,cores=1)

  # This part is for data collection and visualization
  pm.summary(trace).to_csv(output_folder + '/trace_summary.csv')
  pd.DataFrame(np.array(trace['tf'])).to_csv(output_folder + '/tf.csv')
  print(pm.summary(trace))

  #Draw Picture of cvrmse_dist and calculate index
  name_columns = []
  n_columns = n_pred
  for i in range(n_columns):
    name_columns.append('y_pred'+str(i+1))
  y_prediction = pd.DataFrame(np.array(trace['y_pred']),columns=name_columns)
  y_prediction = y_prediction*yc_sd+yc_mean # Scale y_prediction back
  y_prediction.to_csv(output_folder + '/y_pred.csv') # Store y_prediction
  # Calculate the distribution of cvrmse
  cvrmse = 100*np.sqrt(np.sum(np.square(y_prediction-y_true),axis=1)/n_pred)/np.mean(y_true)
  print(np.mean(cvrmse))

  # Calculate the index and store it into csv
  index_cal(y_prediction,y_true).to_csv(output_folder + '/index.csv')

  # Draw pictrue of cvrmse distribution
  plt.subplot(1, 1, 1)
  plt.hist(cvrmse)
  plt.savefig(output_folder + '/cvrmse_dist.pdf')
  plt.close()

  # y_prediction_mean = np.array(pm.summary(trace)['mean'][0:n_pred])*yc_sd+yc_mean
  # cvrmse = 100*np.sqrt(np.sum(np.square(y_prediction_mean-y_true))/len(y_prediction_mean-y_true))/np.mean(y_true)

  #Draw Picture of Prediction_Plot
  y_prediction_mean = np.array(pm.summary(trace)['mean'][0:n_pred])*yc_sd+yc_mean
  y_prediction_975 = np.array(pm.summary(trace)['hpd_97.5'][0:n_pred])*yc_sd+yc_mean
  y_prediction_025 = np.array(pm.summary(trace)['hpd_2.5'][0:n_pred])*yc_sd+yc_mean

  # cvrmse = 100*np.sqrt(np.sum(np.square(y_prediction_mean-y_true))/len(y_prediction_mean-y_true))/np.mean(y_true)
  # print(cvrmse)

  plt.subplot(1, 1, 1)
  # estimated probability
  plt.scatter(x=range(n_pred), y=y_prediction_mean)
  # error bars on the estimate
  plt.vlines(range(n_pred), ymin=y_prediction_025, ymax=y_prediction_975)
  # actual outcomes
  plt.scatter(x=range(n_pred),
             y=y_true, marker='x')

  plt.xlabel('predictor')
  plt.ylabel('outcome')

  plt.savefig(output_folder + '/Prediction_Plot.pdf')

# Resouce file
folder = './1yc total 6tc light equip fan infil chiller boiler 4xc eta-beta=3'
DataComp = np.asarray(pd.read_csv(folder + "/DATACOMP_Single.csv"))
DataField = np.asarray(pd.read_csv(folder + "/DATAFIELD_Single.csv"))[:12,:]
DataPred = np.asarray(pd.read_csv(folder + "/DATAFIELD_Single.csv"))[12:,:]
output_folder = folder
Bayesian_Calibration(DataComp,DataField,DataPred,output_folder)

# folder = './1yc total 4tc light equip fan infil 4xc eta-beta=3'
# DataComp = np.asarray(pd.read_csv(folder + "/DATACOMP_Single.csv"))
# DataField = np.asarray(pd.read_csv(folder + "/DATAFIELD_Single.csv"))[:12,:]
# DataPred = np.asarray(pd.read_csv(folder + "/DATAFIELD_Single.csv"))[12:,:]
# output_folder = folder
# Bayesian_Calibration(DataComp,DataField,DataPred,output_folder)

# folder = './1yc total 5tc light equip fan infil chiller 4xc eta-beta=3'
# DataComp = np.asarray(pd.read_csv(folder + "/DATACOMP_Single.csv"))
# DataField = np.asarray(pd.read_csv(folder + "/DATAFIELD_Single.csv"))[:12,:]
# DataPred = np.asarray(pd.read_csv(folder + "/DATAFIELD_Single.csv"))[12:,:]
# output_folder = folder
# Bayesian_Calibration(DataComp,DataField,DataPred,output_folder)

# DataField_X = RawDataField[:,1:2]
# kmeans = KMeans(n_clusters=16, random_state=0).fit(DataField_X)
# centers = kmeans.cluster_centers_[:,0]

# selected_days = []
# mark=0
# for center in centers:
#     dist = 100
#     for label in range(np.shape(DataField_X)[0]):
#         if dist > np.linalg.norm(center-DataField_X[label]):
#             dist = np.linalg.norm(center-DataField_X[label])
#             mark = label
#     selected_days.append(mark)


# DataComp_label = []
# for i in range(30):
#     for j in range(len(selected_days)):
#         DataComp_label.append(selected_days[j] + i*12)
# DataComp = RawDataComp[DataComp_label]
# DataComp = DataComp[random.sample(range(np.shape(DataComp)[0]), k=300)]