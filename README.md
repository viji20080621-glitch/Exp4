# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 31/10/2026

### AIM:
To implement ARMA model in python.

# Dataset Name: Retail Sales Datasets
# Software Required:
- Jupyter Notebook
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv("retail_sales_dataset.csv")

# Convert Date column
data['Date'] = pd.to_datetime(data['Date'])

# Group daily sales
daily_sales = data.groupby('Date')['Total Amount'].sum()
X = daily_sales.asfreq('D', fill_value=0)

# Declare required variables
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]

# Visualize original data
plt.plot(X)
plt.title('Original Retail Sales Data')
plt.show()

# Plot ACF and PACF
plt.subplot(2, 1, 1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# Fit ARMA(1,1) model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.arparams[0]
theta1_arma11 = arma11_model.maparams[0]

# Simulate ARMA(1,1)
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF
plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

# Fit ARMA(2,2) model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.arparams[0]
phi2_arma22 = arma22_model.arparams[1]

theta1_arma22 = arma22_model.maparams[0]
theta2_arma22 = arma22_model.maparams[1]

# Simulate ARMA(2,2)
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF
plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()

```
# OUTPUT:
# Original dataset
<img width="975" height="467" alt="image" src="https://github.com/user-attachments/assets/b7a095da-9f29-4a01-8152-995e2716c7f6" />

# Original Data ACF 
<img width="1095" height="255" alt="image" src="https://github.com/user-attachments/assets/ff125313-b207-4a3d-a540-cf69e74c9987" />

# Original Data PACF
<img width="1127" height="267" alt="image" src="https://github.com/user-attachments/assets/4d517069-cfa5-4a18-bc32-fc363d553fc9" />

# SIMULATED ARMA(1,1) PROCESS:
<img width="937" height="470" alt="image" src="https://github.com/user-attachments/assets/80b64cde-f086-4246-a3db-f3f00c6e4a0e" />

# Partial Autocorrelation
<img width="940" height="462" alt="image" src="https://github.com/user-attachments/assets/fbb19d70-77d2-4d1e-81c5-7bbffae7c7fa" />

# Autocorrelation
<img width="966" height="470" alt="image" src="https://github.com/user-attachments/assets/63487c67-989e-403e-a514-af676d77970d" />

# SIMULATED ARMA(2,2) PROCESS:
<img width="965" height="465" alt="image" src="https://github.com/user-attachments/assets/5ec2f3d8-0bff-4dbe-8075-f40509a1e8db" />

# Partial Autocorrelation
<img width="970" height="468" alt="image" src="https://github.com/user-attachments/assets/aa883b0b-ef7f-4ca8-a7b5-598f5373aa18" />

# Autocorrelation
<img width="930" height="457" alt="image" src="https://github.com/user-attachments/assets/dad85096-e8be-4996-afe4-22570a543dfa" />

# RESULT:
Thus, a python program is created to fir ARMA Model successfully.
