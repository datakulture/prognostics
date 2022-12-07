import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm
import seaborn as sns
import pickle
import time
from io import BytesIO
import requests
import os
from math import log, exp, sqrt 
from pickle import load
import math
import dash_daq as daq

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error, r2_score 


from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 

image = Image.open('Turbofan.jpg')
st.set_page_config(
	page_title="RUL",
    page_icon=":shark:",
    layout="centered",
    initial_sidebar_state="auto")

Data_url_train = 'https://raw.githubusercontent.com/Ali-Alhamaly/Turbofan_usefull_life_prediction/master/Data/Simulation_Data/train_FD002.txt'
Data_url_test  = 'https://raw.githubusercontent.com/Ali-Alhamaly/Turbofan_usefull_life_prediction/master/Data/Simulation_Data/test_FD002.txt'


@st.cache
def load_data(URL):
	index_columns_names =  ["Engine_No","Time_in_cycles"] 
	operational_settings_columns_names = ['Altitude','Mach_number','TRA'] 
	sensor_measure_columns_names = ['T2','T24','T30','T50', 'P2','P15','P30', 'Nf','Nc','epr','Ps30','phi','NRf','NRc','BPR','fuel_air_ratio','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']
	input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names
	df= pd.read_csv(URL,delim_whitespace=True,names=input_file_column_names,error_bad_lines=False) 
	# convert killo feet to meters
	df['Altitude']= df['Altitude'].apply(lambda x: x*1000*0.3048)
	#convert temperature Rankine to Kelvin
	df['T2'] = df['T2'].apply(lambda x: x*0.555556)
	#convert temperature Rankine to Kelvin
	df['T24'] = df['T24'].apply(lambda x: x*0.555556)
	#convert temperature Rankine to Kelvin 
	df['T30'] = df['T30'].apply(lambda x: x*0.555556)
	#convert temperature Rankine to Kelvin
	df['T50'] = df['T50'].apply(lambda x: x*0.555556)
	# convert pressure psia to bar
	df['P2']  = df['P2'].apply(lambda x: x*0.0689476)
	# convert pressure psia to bar
	df['P15']  = df['P15'].apply(lambda x: x*0.0689476)
	 # convert pressurepsia to bar
	df['P30']  = df['P30'].apply(lambda x: x*0.0689476)
	# convert pressure to bar
	df['Ps30']  = df['Ps30'].apply(lambda x: x*0.0689476) 
	return df


@st.cache
def RUL(df):
    rul = pd.DataFrame(df.groupby('Engine_No')['Time_in_cycles'].max()).reset_index() # Data Labeling - generate column RUL
    rul.columns = ['Engine_No', 'max_No_cycles'] 
    df= df.merge(rul, on=['Engine_No'], how='left') 
    df['RUL'] = df['max_No_cycles'] - df['Time_in_cycles'] 
    df.drop('max_No_cycles', inplace=True, axis=1)
    return df 

def load_homepage() -> None:
	st.title('Estimate Remaining Useful Life of Turbofan engines') 
	st.image(image, caption='Photographer: Luke Sharrett/Bloomberg', use_column_width=True) 
	st.write("""
		One of the main challenges in the aviation industry is to reduce maintenance costs and reduce downtime of machines by maintaining or improving safety standard. A large per cent of these costly delays are a result of unplanned maintenance such as when an aircraft has an abnormal behaviour on the previous flight, creating an operational disruption and even requiring an aircraft change.

		Since you don't know when failure will occur, you have to be conservative in your planning, especially if you're operating safety-critical equipment like engines. But by scheduling maintenance very early, you're wasting machine life or spare part life that is still usable, and this adds to costs to the owner.

		However, if you can predict when machine failure will occur, you can schedule maintenance right before it. By implementing predictive maintenance we can minimize unnecessarily scheduling maintenance and production hours lost during maintenance (improve the overall availability of equipment) and reduce the cost of spare parts and its consumables during the maintenance process.
		
		In order to develop an algorithm that predicts the breakdown of a piece of equipment within a given time window (typically some number of days), we require enough historical data that allows us to capture information about events leading to failure.""") 
	st.subheader('About Data set:')
	st.write("""
		In this post, we are using the C-MAPSS dataset, which is engine degradation simulation that was carried out at NASA by using C-MAPSS simulation software. C-MAPSS has created four different data sets simulated under different combinations of operational conditions and fault modes.
		
		Data sets consist of three operational settings and 21 sensor measurements (temperature, pressure, fan speed, etc.) for several engines and for every cycle of their lifetime. 

		*The engine is operating normally at the start of each time series and develops a fault at some point during the series.**In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends sometime prior to system failure**.*""")
	st.subheader("Problem statement:")
	st.write("The goal of predictive maintenance is to predict at the time t, using the data up to that time, whether the equipment will fail in the near future.")
	st.write("This problem can be formulated in two ways:")
	st.write("**Classification:** we aim to predict the probability that the equipment will fail within a pre-specified time window.") 
	st.write("**Regression:** A regression-based approach which aims to estimate the remaining time to the end of the equipment's useful life of an engine.")
	st.write("In this blog, we'll focus on the second dataset (FD002) in which all engines develop some fault with six operating conditions.") 


def EDA():
	st.title('Data Exploration') 
	if st.checkbox("Show Train dataset"):
		load_stats =st.text("Loading Train data..............")
		df_train = load_data(Data_url_train)
		load_stats.text("Loading Train data....done!")
		st.subheader('Train data')
		st.text("All values are converted in SI system")
		st.write(df_train.head(10)) 
	if st.checkbox("Show Test dataset"):
		load_stats.text("Loading Test data..............") 
		df_Test = load_data(Data_url_test)
		load_stats.success(" Done ")
		st.subheader('Test data')
		st.text("All values are converted in SI system")
		st.write(df_Test.head(10)) 

	st.write("""Before we start EDA, we need to compute a target variable for this data set which is RUL.""")
	st.write("""(RUL)Remaining Useful Life(the length of time a machine is likely to operate before it requires repair or replacement.""")
	st.write("""As mentioned above, the definition, 

    RUL = a total number of the life cycle engine - Current engine life cycle """)
	st.write("""To calculate RUL in our training data set. we just did group by engine number(max number of cycles) and present engine life cycle.""")
	st.write("""For test data, the time series ends sometime prior to system failure, after that we don't have data and we have ground-truth data (The true remaining cycles for each engine in the testing data).

Example: let's consider the engine one life cycle data. In a training data life cycle of engine start from 1 and end 223. In test data, the engine runs some cycle like 120, we need to find out how many cycles still remain. 

To calculate RUL in our test data set, we did the same thing as training data by adding ground-truth data value to test data.""")
	df_train = load_data(Data_url_train)
	train = RUL(df_train)
	if st.checkbox("Show code"):
		st.write('code to calculate RUL')
		code = '''rul = pd.DataFrame(df_train.groupby('Engine_No')['Time_in_cycles'].max()).reset_index() # Data Labeling - generate column RUL
		  rul.columns = ['Engine_No', 'max_No_cycles']
		  train = df_train.merge(rul, on=['Engine_No'], how='left')
		  train['RUL'] = train['max_No_cycles'] - train['Time_in_cycles'] 
                                ####################################
		  truth_df = pd.read_csv('https://raw.githubusercontent.com/Ali-Alhamaly/Turbofan_usefull_life_prediction/master/Data/Simulation_Data/RUL_FD002.txt',delim_whitespace=True, header=None)
		  truth_df.columns = ['RUL'] 

		  # generate column max for test data 
		  rul = pd.DataFrame(df_Test.groupby('Engine_No')['Time_in_cycles'].max()).reset_index()
		  rul.columns = ['Engine_No', 'max_No_cycles'] 
		  truth_df['Engine_No'] = truth_df.index + 1
		  truth_df['max_No_cycles'] = rul['max_No_cycles'] + truth_df['RUL']
		  truth_df.drop('RUL', axis=1, inplace=True)
		  # generate RUL for test data
		  Test = df_Test.merge(truth_df, on=['Engine_No'], how='left')
		  Test['RUL'] = Test['max_No_cycles'] - Test['Time_in_cycles']'''
		st.code(code, language='python') 
	st.write("Data with RUL calculation") 
	st.write(train.head(5)) 
	st.write('''Since the degradation in a system will generally remain negligible until after some period of the operation time, the early and higher RUL values are probably unreasonable. We can't really say anything about the RUL before that point because we have no information about the initial wear and tear.''')
	if st.checkbox("Show Sketch of the RUL of Engines"):
		st.write('''Instead of having our RUL decline linearly, we define our RUL to start out as a constant and only decline linearly after some time''')
		#rul_clip_limit
		rul_clip_limit = 125
		y_train= train[train['Engine_No']==10].RUL.reset_index(drop=True) 
		y_train_clip= y_train.clip(upper =rul_clip_limit) 
		fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
		axes[0].plot(y_train)
		axes[0].set_title('Engine RUL before the Clipping')
		axes[1].plot(y_train_clip)
		axes[1].set_title('Engine RUL after the Clipping')
		st.pyplot(fig)
	st.header('**Exploratory Data Analysis:**') 
	st.bar_chart(train.Engine_No.value_counts(),use_container_width=True) 
	st.write("- Engines have different life durations.The average working time in train data is 206 cycles with a minimum of 128 and a maximum of 378.")
	st.write("- 96% of engines are within 300 life cycle and 4% only engines lived above 300 life cycles, this engines we consider as an extreme case.")
	st.subheader('**Operational Setting:**') 
	option_1 = st.selectbox('What column do you want to display',['Altitude', 'Mach_number', 'TRA'])
	#st.bar_chart(train[option_1].value_counts(),use_container_width=True) 

	fig = px.scatter(train.reset_index(),x='index',y = str(option_1))
	st.plotly_chart(fig)
	
	st.write("""As plots illustrate, all engines are operated six different altitude conditions with a range of (0–12000M.) and with 5 different operating speed(Mach number) of (0–0.84), with two throttle position (TRA) (60–100).""")
	st.subheader('**sensors data:**') 
	option = st.multiselect('What column do you want to display',train.columns.difference(['Engine_No', 'Time_in_cycles', 'Altitude', 'Mach_number', 'TRA']))
	@st.cache
	def sensor_data(Engine_No=1):
		df = train[train['Engine_No'] == 1]
		min_max= MinMaxScaler()
		cols = df.columns.difference(['Engine_No', 'Time_in_cycles', 'Altitude', 'Mach_number', 'TRA'])
		norm_train_df = pd.DataFrame(min_max.fit_transform(df[cols]), columns=cols, index=df.index)
		join_df = df[df.columns.difference(cols)].join(norm_train_df)
		Train= join_df.reindex(columns = df.columns) 
		return df
	df = sensor_data()
	st.line_chart(df[option]) 
	st.write("- sensors data remain same throughout engine's life.")
	st.write("- we understand that sensors data T2,T24,T30,T50,P2,P15,P30 highly correlated with one another (Based on Temperature and pressure graph look like sames).")
	st.write("- sensors Nf and Nc also correlated and also W31 and W32")
	st.write('- As we can see, all the sensors data is higly correlated. even though data is highly correlated we are keeping these features.') 
	st.write("Let's plot PDF individual sensors for any given engines.") 
	option_2 = st.multiselect('which engine do you want for comparison',train.Engine_No.unique())
	# Radio Buttons
	status = st.radio("Plot PDF of below sensors data ",("Pressure sensors","Temperature sensors","Speed sensors","Other parameters"))
	if status =="Pressure sensors":
		option_3= st.multiselect('Select from Pressure sensors',['P2','P15','P30','Ps30'],['P2','P15'])	
	elif status == "Temperature sensors":
		option_3= st.multiselect('Select from temperature sensors',['T2','T24','T30','T50'],['T2','T50'])
	elif status == "Speed sensors":
		option_3= st.multiselect('Select from Speed sensors',['Nf','Nc','NRf','NRc','Nf_dmd','PCNfR_dmd','W31','W32'],['W31','W32'])
	else:
		option_3= st.multiselect('Select from Other parameters',['epr','phi','BPR','fuel_air_ratio','htBleed'],['BPR','fuel_air_ratio'])
	df= train.loc[train['Engine_No'].isin(option_2)].reset_index(drop=True)
	fig = sns.pairplot(df, vars=option_3, hue="Engine_No")
	st.pyplot(fig)
	for i in option_2:
		st.write('Engine number is',i, ', total number of engine cycle(RUL) is', df[df['Engine_No']==i].shape[0])
	st.write("when we are comparing this engines sensors data, we couldn't differentiate between engines by looking into PDF plots for this data set")

def FE():
	st.title('Feature engineering')
	st.write('Based on the domain knowledge we have come up with 40 new features which can improve model performance. These features are performance parameters of turbofan engines like, Thrust, Propulsion efficiency, Thermal efficiency,Turbine Entry Temperature(TET),etc.....')
	image = Image.open('C-MAPSS.jpg')
	st.image(image, caption='From C-MAPSS paper', use_column_width=True) 
	st.write("Schematic diagram showing the components, station numbers and sensor data taken from of the C-MAPSS engine. A brief description some of the major modules is summarized below.")
	st.write("**Exhaust Gas Temperature (EGT)(T90):** expressed in Kelvin, is the temperature at the engine exhaust and a measure of an engine’s efficiency in producing its design level thrust; the higher the EGT the more wear and deterioration affect an engine. High EGT can be an indication of degraded engine performance. An exceedance in EGT limits can lead to immediate damages of engine parts and/or a life reduction of engine parts. With this in mind it then becomes absolutely important to keep the EGT as low as possible for as long as possible.")
	st.write ("**Turbine Inlet Temperature(T41):** The general trend is that raising the turbine inlet temperature increases the specific thrust of the engines with a small increase in fuel consumption rate. However, high temperatures can damage the turbine, as the blades are under large centrifugal stresses and materials are weaker at high temperature.")
	st.write("**Thermal efficiency** is a prime factor in turbine performance.The three most important factors affecting the thermal efficiency are turbine inlet temperature(T41), compression ratio(P30/P21), and the component efficiencies of the compressor and turbine. Other factors that affect thermal efficiency are compressor inlet temperature(P21) and combustion efficiency.Maximum thermal efficiency can be obtained by maintaining the highest possible exhaust temperature.since engine life is greatly reduced at high turbine inlet temperatures, the operator should not exceed the exhaust temperatures specified for continuous operation.")
	link_1 = "All these features are created using given sensor data with some assumptions. Please ref to [[1]](http://www.aircraftmonitor.com/uploads/1/5/9/9/15993320/engine_mx_concepts_for_financiers___v2.pdf),[[2]](https://www.flight-mechanic.com/gas-turbine-engine-performance/)"
	st.markdown(link_1, unsafe_allow_html=True)

	st.write('**Improvement:**')
	st.write("- From EDA we understand that the data set has different operating conditions, so we can create clustering and do labeling on top of that and create onehotencoder but this only work on FD002 and FD004 datasets")
	st.write("- If possible we can create feature engineering with autoencoder.")
	st.write('- If we can remove noise from data, we can achieve good results. Let’s try noise reduction effectively by using the autoencoder or other noise reduction techniques')



def MS():
	st.title("Model Building")
	st.write("As mentioned early in home page we are going to solve both regression and classification.")
	st.write("After completing feature engineering, we have got in total 63 feature and applied random forest algorithm to select features and finally we have 23 features.")
	st.write("We have 4 different data set with different operating conditions. Please Select data set below.") 

	data_set_type= st.selectbox("Please choose a data set type", ("FD001", "FD002","FD003", "FD004"))

	if data_set_type == "FD001":
		url_train = 'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/output/FD001/Feature_Engg_Train_F0001.csv'
		url_test =  'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/output/FD001/Feature_Engg_Test_F0001.csv'
	elif data_set_type == "FD002":
		url_train = 'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/output/FD002/Feature_Engg_Train_F0002.csv'
		url_test =  'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/output/FD002/Feature_Engg_Test_F0002.csv'

	elif data_set_type == "FD003":
		url_train = 'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/output/FD003/Feature_Engg_Train_F0003.csv'
		url_test =  'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/output/FD003/Feature_Engg_Test_F0003.csv'
	else:
		url_train = 'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/output/FD004/Feature_Engg_Train_F0004.csv'
		url_test =  'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/output/FD004/Feature_Engg_Test_F0004.csv'

	st.subheader("Scaling:")
	st.write("Sklearns’ MinMaxScaler can be used to create a scaler fitted to our train data. The default settings create a scaler to scale our training features between 0–1. The scaler is then applied to both our x_train and x_test set.")
	imp_col = ['RUL','Engine_No','T50','Nf','Ps30','phi','NRf','NRc','BPR','W31','W32','T48','T41','T90','Ve','EGT_margin','Nc/Nf','PCNcRdmd ','M_cold','W_f','Thrust','Fan_thrust','core_thrust','TSFC','Thermal_efficiency']
	@st.cache(allow_output_mutation=True)
	def load_feature(URL):
		df= pd.read_csv(URL,usecols =imp_col)
		return df 
	Train_feature = load_feature(url_train) 
	Test_feature= load_feature(url_test) 
	period=30
	Train_feature['label'] = Train_feature['RUL'].apply(lambda x: 1 if x <= period else 0) 
	Test_feature['label'] =  Test_feature['RUL'].apply(lambda x: 1 if x <= period else 0) 

	min_max= MinMaxScaler()
	cols = Train_feature.columns.difference(['RUL','label','Engine_No']) 
	norm_train_df = pd.DataFrame(min_max.fit_transform(Train_feature[cols]), columns=cols, index=Train_feature.index)
	join_df = Train_feature[Train_feature.columns.difference(cols)].join(norm_train_df)
	Train= join_df.reindex(columns = Train_feature.columns) 

	norm_train_df = pd.DataFrame(min_max.transform(Test_feature[cols]), columns=cols, index=Test_feature.index)
	join_df = Test_feature[Test_feature.columns.difference(cols)].join(norm_train_df)
	Test= join_df.reindex(columns = Test_feature.columns)
	st.write("Shape of train data :",Train.shape) 
	st.write("Shape of test data :",Test.shape)
	st.subheader("Data processing:")
	st.write("By default we are having Train and Test datasets. We keep Train data for train and split test data into cross_validation and test data, we pick the last sequence data points as test data,remaining data points as cross_validation.")

	st.write("We have taken RUL and Label columns as dependent variable, remaining 23 columns as independent variable and remaining one columns is Engine_No column.")

	st.write("For classification approach, generating the label by ourself in this way, from 0 to 30 cycles, we’ve labeled as 1(fail) and rest (>30) as 0(Not fail). For regression problem we keep RUL columns without any changes.")
	st.write(Train.head())

	@st.cache 
	def splite_data(sequence_length,data,cross_val=True):
		val_data=[]
		for id in data['Engine_No'].unique():
			if len(data[data['Engine_No']==id])>= sequence_length:
				if cross_val:
					# Remaining data points as cross_validation(other than last sequence data points).
					val_data.append(data[data['Engine_No']==id][:-sequence_length])
				else:
					# We pick the last sequence data points as test data
					val_data.append(Test[Test['Engine_No']==id][-sequence_length:])
			else:
				False
		return pd.concat(val_data)
	st.write("After split data, we process data for machine learning and deep learing algorithm little differently which is followed below.")
	#This function plots the confusion matrices given y_i, y_i_hat.

	st.subheader("**Choose the algorithm**")
	ML_type= st.selectbox("Please choose a Model Type", ("Machine learning", "Deep learning"))
	@st.cache
	def RUL_line(Y_true, Y_pred):
		dataset = pd.DataFrame({'True_RUL':Y_true, 'Predic_RUL': Y_pred}, columns=['True_RUL', 'Predic_RUL'])
		return dataset
	def plot_confusion_matrix(test_x, test_y, clf):
		pred_y = sig_clf.predict(test_x)
		st.write("-"*20, "Confusion matrix", "-"*20)
		cm = confusion_matrix(test_y,pred_y)
		fig, ax = plt.subplots(figsize=(3,3))
		ax = sns.heatmap(cm,annot=True, cmap='Blues')
		fig.suptitle('Confusion matrix', fontsize=5)
		plt.xlabel('Predicted Class', fontsize=5)
		plt.ylabel('Original Class', fontsize=5)
		st.pyplot(fig) 
        # calculating the number of data points that are misclassified
		st.write("Number of mis-classified points :", (np.count_nonzero((pred_y- test_y))/test_y.shape[0])*100)
		precision, recall, fscore, support = score(test_y,pred_y)
		st.write('precision:', precision[0])
		st.write('recall:',recall[0])
		st.write('fscore:',fscore[0])
		st.write('Negative class labels:',support[0],'positive class labels:',support[1])

	if ML_type == "Machine learning":
		Algo_type = st.selectbox("Please choose a Problem Type", ("Classification", "Regression"))
		
		if Algo_type == 'Classification':
			st.write("For machine learning algorithm we send data to algorithm as a individual data point.")
			# cross_val is other than last sequence data points (test data 41214 = cross_val(40966)+test(248))
			cross =     splite_data(1,Test,cross_val=True) 
			# we have 248 engine data,# We pick the last sequence data points as test data  
			Test_data = splite_data(1,Test,cross_val= False)

			y_train =  Train['label'] 
			x_train =  Train.drop(['RUL','label','Engine_No'],axis = 1)

			y_cross_val =  cross['label'] 
			x_cross_val =  cross.drop(['RUL','label','Engine_No'],axis = 1)

			y_test =  Test_data['label'].reset_index(drop=True)
			x_test =  Test_data.drop(['RUL','label','Engine_No'],axis = 1).reset_index(drop=True) 

			st.write("Shape of Train data after split:",x_train.shape) 
			st.write("Shape of Train labeling data :",y_train.shape)
			st.write("Shape of Cross validation data after split:",x_cross_val .shape) 
			st.write("Shape of Cross validation labeling data :",y_cross_val.shape) 
			st.write("Shape of  Test data after split:",x_test.shape) 
			st.write("Shape of Test labeling data :",y_test.shape)
			if st.checkbox("Show code for ML data processing"):
				st.write('code for to Data processing:')
				data_code = '''
				def splite_data(sequence_length,data,cross_val=True):
					val_data=[]
					# we iterating through each engine number 
					for id in data['Engine_No'].unique():
						if len(data[data['Engine_No']==id])>= sequence_length:
							if cross_val:
								# Remaining data points as cross_validation(other than last sequence data points).
								val_data.append(data[data['Engine_No']==id][:-sequence_length])
							else:
								# We pick the last sequence data points as test data
								val_data.append(Test[Test['Engine_No']==id][-sequence_length:]) 
						else:
							False
					return pd.concat(val_data)'''
				st.code(data_code, language='python') 
			select_model = st.selectbox("Please choose a Model from list", ("Logistic regression", "Linear SVM","RF Classifier"))
			if select_model == "Logistic regression":
				st.write("Let’s train",select_model,"on", data_set_type, "data set") 
				st.sidebar.markdown("----")
				C= st.sidebar.select_slider('select C parameter to controls the penality strength',options=[10 ** x for x in range(-7, 1)]) 
				st.sidebar.write("Select C values is", C)
				Regue = st.sidebar.selectbox("Select regularization",['none', 'l1', 'l2', 'elasticnet']) 
				st.sidebar.write("Select regularizer is",Regue)
				#sol = st.sidebar.selectbox("Select solvers to convergence ",['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
				@st.cache
				def train_model(C=1e-06,Regue = "l2"):
					clf = SGDClassifier(alpha=C, penalty= Regue, loss='log', random_state=42)
					clf.fit(x_train, y_train)
					sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
					sig_clf.fit(x_train, y_train) 
					return sig_clf
				st.sidebar.write("Train",select_model)
				if st.sidebar.button("Train model"):
					with st.spinner("Training ongoing......"):
						sig_clf= train_model(C=C,Regue = Regue)
				else:
					sig_clf= train_model(C=1e-06,Regue = "l2") 
					sig_clf.fit(x_train, y_train)
				predict_y = sig_clf.predict_proba(x_train) 
				st.write("The train log loss is:",log_loss(y_train, predict_y, labels=[0,1], eps=1e-15))
				
				predict_y = sig_clf.predict_proba(x_cross_val) 
				st.write("The cross validation log loss is:",log_loss(y_cross_val, predict_y, labels=[0,1], eps=1e-15))

				predict_y = sig_clf.predict_proba(x_test)
				st.write("The test log loss is:",log_loss(y_test, predict_y, labels=[0,1], eps=1e-15))
				plot_confusion_matrix(x_test,y_test,sig_clf)

			elif select_model == "Linear SVM": 
				st.write("Let’s train",select_model,"on", data_set_type, "data set") 
				st.sidebar.markdown("----")
				C= st.sidebar.select_slider('select C parameter to controls the penality strength',options=[10 ** x for x in range(-7, 3)]) 
				st.sidebar.write("Select C values is", C)
				Regue = st.sidebar.selectbox("Select regularization",['none', 'l1', 'l2', 'elasticnet']) 
				st.sidebar.write("Select regularizer is",Regue)
				#sol = st.sidebar.selectbox("Select solvers to convergence ",['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
				@st.cache(allow_output_mutation=True)
				def train_model(C=1e-06,Regue = "l2"):
					clf = SGDClassifier(alpha=C, penalty= Regue, loss='hinge', random_state=42)
					clf.fit(x_train, y_train)
					sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
					sig_clf.fit(x_train, y_train) 
					return sig_clf
				st.sidebar.write("Train",select_model)
				if st.sidebar.button("Train model"):
					with st.spinner("Training ongoing......"):
						sig_clf= train_model(C=C,Regue = Regue)
				else:
					sig_clf= train_model(C=1e-06,Regue = "l2") 
					sig_clf.fit(x_train, y_train)
				predict_y = sig_clf.predict_proba(x_train) 
				st.write("The train log loss is:",log_loss(y_train, predict_y, labels=[0,1], eps=1e-15))
				
				predict_y = sig_clf.predict_proba(x_cross_val) 
				st.write("The cross validation log loss is:",log_loss(y_cross_val, predict_y, labels=[0,1], eps=1e-15))

				predict_y = sig_clf.predict_proba(x_test)
				st.write("The test log loss is:",log_loss(y_test, predict_y, labels=[0,1], eps=1e-15))
				plot_confusion_matrix(x_test,y_test,sig_clf) 

			else:
				st.write("Let’s train",select_model,"on", data_set_type, "data set") 
				st.sidebar.markdown("----") 

				n_estimators = st.sidebar.select_slider('select number of estimators',options=[5,10,15,20,25,30,35,40,45,50,75,100,200,500,1000])
				st.sidebar.write("Selected number of estimators are ",n_estimators)
				max_depth =    st.sidebar.select_slider('select max_depth of tree',options=[5,10,15,20,25,30,35]) 
				st.sidebar.write("Selected tree depth is ",max_depth)
				@st.cache(allow_output_mutation=True)
				def rf_model(n_estimators=500,max_depth=10):
					clf = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=max_depth, random_state=42, n_jobs=-1)
					clf.fit(x_train,y_train)
					sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
					sig_clf.fit(x_train,y_train)
					return sig_clf
				st.sidebar.write("Train",select_model)
				if st.sidebar.button("Train model"):
					with st.spinner("Training ongoing......"):
						sig_clf= rf_model(n_estimators=n_estimators,max_depth=max_depth)
				else:
					sig_clf= rf_model(n_estimators=500,max_depth=10) 
					sig_clf.fit(x_train,y_train)
				predict_y = sig_clf.predict_proba(x_train) 
				st.write("The train log loss is:",log_loss(y_train, predict_y, labels=[0,1], eps=1e-15))
				
				predict_y = sig_clf.predict_proba(x_cross_val) 
				st.write("The cross validation log loss is:",log_loss(y_cross_val, predict_y, labels=[0,1], eps=1e-15))

				predict_y = sig_clf.predict_proba(x_test)
				st.write("The test log loss is:",log_loss(y_test, predict_y, labels=[0,1], eps=1e-15))
				plot_confusion_matrix(x_test,y_test,sig_clf) 
		else:
			st.write("For machine learning algorithm we send data to algorithm as a individual data point.")
			# cross_val is other than last sequence data points (test data 41214 = cross_val(40966)+test(248))
			cross =     splite_data(1,Test,cross_val=True) 
			# we have 248 engine data,# We pick the last sequence data points as test data  
			Test_data = splite_data(1,Test,cross_val= False)

			y_train =  Train['RUL'] 
			x_train =  Train.drop(['RUL','label','Engine_No'],axis = 1)

			y_cross_val =  cross['RUL'] 
			x_cross_val =  cross.drop(['RUL','label','Engine_No'],axis = 1)

			## Clipped RUL
			y_train_clipped = y_train.clip(upper=125) 
			y_cross_val_clipped = y_cross_val.clip(upper=125) 

			y_test =  Test_data['RUL'].reset_index(drop=True)
			x_test =  Test_data.drop(['RUL','label','Engine_No'],axis = 1).reset_index(drop=True) 

			st.write("Shape of Train data after split:",x_train.shape) 
			st.write("Shape of Train labeling data :",y_train_clipped.shape)
			st.write("Shape of Cross validation data after split:",x_cross_val .shape) 
			st.write("Shape of Cross validation labeling data :",y_cross_val_clipped.shape) 
			st.write("Shape of  Test data after split:",x_test.shape) 
			st.write("Shape of Test labeling data :",y_test.shape)
			if st.checkbox("Show code for ML data processing"):
				st.write('code for to Data processing:')
				data_code = '''
				def splite_data(sequence_length,data,cross_val=True):
					val_data=[]
					# we iterating through each engine number 
					for id in data['Engine_No'].unique():
						if len(data[data['Engine_No']==id])>= sequence_length:
							if cross_val:
								# Remaining data points as cross_validation(other than last sequence data points).
								val_data.append(data[data['Engine_No']==id][:-sequence_length])
							else:
								# We pick the last sequence data points as test data
								val_data.append(Test[Test['Engine_No']==id][-sequence_length:]) 
						else:
							False
					return pd.concat(val_data)'''
				st.code(data_code, language='python') 
			select_model = st.selectbox("Please choose a Model from list", ("Linear regression","RF regression")) 



			# performance matrix
			def evaluate(y_true, y_hat,data_type ="Train"):
				mse = mean_squared_error(y_true, y_hat)
				rmse = np.sqrt(mse)
				variance = r2_score(y_true, y_hat)
				return st.write("{} data RMSE: {} and R2: {} ".format(data_type, rmse,variance))
			if select_model == "Linear regression":
				st.write("Let’s train baseline model,",select_model,"on", data_set_type, "data set") 
				lm = LinearRegression()
				lm.fit(x_train, y_train_clipped)

				# predict and evaluate
				y_hat_train = lm.predict(x_train)
				evaluate(y_train_clipped, y_hat_train,"Train")

				y_hat_cross = lm.predict(x_cross_val)
				evaluate(y_cross_val_clipped, y_hat_cross,'Cross validation')

				y_hat_cross = lm.predict(x_test)
				evaluate(y_test, y_hat_cross,"Test") 
				st.line_chart (RUL_line(y_test, y_hat_cross)) 

			else:
				st.write("Let’s train model,",select_model,"on", data_set_type, "data set")
				st.sidebar.markdown("----") 

				n_estimators = st.sidebar.select_slider('select number of estimators',options=[5,10,15,20,25,30,35,40,45,50,75,100,200,500,1000])
				st.sidebar.write("Selected number of estimators are ",n_estimators)
				max_depth =    st.sidebar.select_slider('select max_depth of tree',options=[5,10,15,20,25,30,35])
				st.sidebar.write("Selected tree depth is ",max_depth)
				@st.cache(allow_output_mutation=True)
				def rf_reg_model(n_estimators=30, max_depth=15):
					clf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, max_features='sqrt', min_samples_split=5, random_state=42, n_jobs=-1)
					clf.fit(x_train,y_train)
					return clf
				st.sidebar.write("Train",select_model) 
				if st.sidebar.button("Train model"):
					with st.spinner("Training ongoing......"):
						clf= rf_reg_model(n_estimators=n_estimators, max_depth=max_depth)
						# predict and evaluate
						y_hat_train = clf.predict(x_train)
						evaluate(y_train_clipped, y_hat_train,"Train")
						y_hat_cross = clf.predict(x_cross_val)
						evaluate(y_cross_val_clipped, y_hat_cross,'Cross validation')
						y_hat_cross = clf.predict(x_test)
						evaluate(y_test, y_hat_cross,"Test") 
						st.line_chart (RUL_line(y_test, y_hat_cross))
				else:
					# Run model with best hyper paramaters
					clf_hy = RandomForestRegressor(n_estimators=30,max_depth=15, max_features='sqrt', min_samples_split=5, random_state=42, n_jobs=-1)
					clf_hy.fit(x_train,y_train)
					# predict and evaluate
					y_hat_train = clf_hy.predict(x_train)
					evaluate(y_train_clipped, y_hat_train,"Train")

					y_hat_cross = clf_hy.predict(x_cross_val)
					evaluate(y_cross_val_clipped, y_hat_cross,'Cross validation')

					y_hat_cross = clf_hy.predict(x_test)
					evaluate(y_test, y_hat_cross,"Test")
					st.line_chart (RUL_line(y_test, y_hat_cross)) 

	else:
		def plot_confusion_matrix_dp(input,input_lable,model):
			pred_y  = np.argmax(model.predict(input), axis=-1) 
			test_y = input_lable 
			st.write("-"*20, "Confusion matrix", "-"*20)
			cm = confusion_matrix(test_y,pred_y)
			fig, ax = plt.subplots(figsize=(3,3))
			ax = sns.heatmap(cm,annot=True, cmap='Blues')
			fig.suptitle('Confusion matrix', fontsize=5)
			plt.xlabel('Predicted Class', fontsize=5)
			plt.ylabel('Original Class', fontsize=5)
			st.pyplot(fig) 
			# calculating the number of data points that are misclassified
			st.write("Number of mis-classified points :", (np.count_nonzero((pred_y- test_y))/test_y.shape[0]))
			precision, recall, fscore, support = score(test_y,pred_y)
			st.write('precision:', precision[0])
			st.write('recall:',recall[0])
			st.write('fscore:',fscore[0])
			st.write('Negative class labels:',support[0],'positive class labels:',support[1])


def model_predi():
	st.title("Model Prediction")

	st.write("Let’s predict how models are performing by giving unseen data.")
	st.sidebar.markdown("---")
	data_set_type= st.sidebar.selectbox("Please choose a Data set", ("FD001", "FD002","FD003", "FD004"))

	@st.cache
	def imp_features(data_path): 
		index_columns_names =  ["Engine_No","Time_in_cycles"] 
		operational_settings_columns_names = ['Altitude','Mach_number','TRA'] 
		sensor_measure_columns_names = ['T2','T24','T30', 'T50', 'P2','P15','P30', 'Nf','Nc','epr','Ps30','phi','NRf','NRc','BPR','fuel_air_ratio','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']
		input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names
		machine_df= pd.read_csv(data_path,delim_whitespace=True,names=input_file_column_names) 
		imp_col = ['Engine_No','T50','Nf','Ps30','phi','NRf','NRc','BPR','W31','W32','T48','T41','T90','Ve','EGT_margin','Nc/Nf','PCNcRdmd ','M_cold','W_f','Thrust','Fan_thrust','core_thrust','TSFC','Thermal_efficiency']
		###############################################
		cp_gas = 1147 #specific heat of gas
		cp_air = 1005 # #specific heat of air
		transmission_efficiency = 0.985 # of high pressure compressor 
		gamma_gas = 1.33
		gamma_air = 1.4    #  specific heats of air
		nozzle_efficiency =0.95  # efficiency of nozzle
		fan_efficiency =0.99    #  efficiency of turbo fan
		nozzle_efficiency = 0.995 # nozzle fan efficiency 
		R = 287  # gas constant
		machine_df['Altitude']= machine_df['Altitude'].apply(lambda x: x*1000*0.3048) # convert killo feet to meters 
		machine_df['T2'] =  machine_df['T2'].apply(lambda x: x*0.555556)  #convert temperature Rankine to Kelvin 
		machine_df['P2']  = machine_df['P2'].apply(lambda x: x*0.0689476) # convert pressure psia to bar
		machine_df['T24'] = machine_df['T24'].apply(lambda x: x*0.555556) #convert temperature Rankine to Kelvin
		machine_df['T30'] = machine_df['T30'].apply(lambda x: x*0.555556) #convert temperature Rankine to Kelvin
		machine_df['T50'] = machine_df['T50'].apply(lambda x: x*0.555556) #convert temperature Rankine to Kelvin 
		machine_df['Ps30'] = machine_df['Ps30'].apply(lambda x: x*0.0689476) # convert pressure to bar 
		#######################################################
		#######################################################
		# calc High pressure turbine outlet temperature 
		def HPT_outlet(T24,T2,T50,BPR):
			T48= T50+(cp_air*(T24-T2)*(BPR+1))/(cp_gas*transmission_efficiency)
			return T48 
		machine_df['T48'] = machine_df.apply(lambda row :HPT_outlet(row['T24'],row['T2'],row['T50'],row['BPR']),axis= 1) 
		######################################################
		#Turbine entry temperature 
		def turbin_inlet(T48,T30,T24):
			T41 = T48+(cp_air*(T30-T24))/(cp_gas*transmission_efficiency)
			return T41 
		machine_df['T41']= machine_df.apply(lambda row: turbin_inlet(row['T48'],row['T30'],row['T24']),axis = 1) 
		machine_df['P50'] = machine_df['epr']*machine_df['P2']
		######################################################
		# calculate pressure,temperature and density of an aircraft at its flying altitudes.
		def isa(altitude):
			# troposphere
			if altitude<=11000:
				if altitude == 0:
					temperature =    228.15
					pressure    =    1.01325
					density    =     1.225 
				else:
					temperature = 288.15 - 0.0065 * altitude
					pressure =   1.01325 * (temperature/288.15)**5.25588
					density  =   (pressure/(287.05287*temperature))*10**5
			# stratosphere
			elif altitude <=20000:
				temperature =        288.15 - 0.0065 *11000
				pressure_11000    =  1.01325 * (temperature/288.15)**5.25588
				density_11000     =  (pressure_11000/(287.05287*temperature))*10**5
				ratio             =  exp((-9.80665*(altitude-11000))/(287.05287*temperature))
				pressure          =  pressure_11000*ratio
				density           =  density_11000*ratio
			else:
				raise ValueError('altitude out of range [0-20000m]')

			return  pressure,temperature,density
		machine_df[['Atm_pressure','Atm_temp','Atm_density']] = machine_df.apply(lambda row: isa(row['Altitude']),axis=1,result_type="expand") 
		################################################
		def aircraft_velocity(Mach_number,Atm_temp):
			V_a= Mach_number*(sqrt(gamma_air*R*Atm_temp))
			return V_a
		machine_df['Va'] = machine_df.apply(lambda row:aircraft_velocity(row['Mach_number'],row['Atm_temp']),axis=1)
		################################################
		def EGT(P50,T50,atm_pressure):
			R = ((gamma_gas-1)*cp_gas)/gamma_gas
			nozzle_pressure_ratio = P50/atm_pressure  # Total temperature at LPT outlet/atmospheric pressure 
			# check nozzle choked condition or not 
			choked_ration =  (1- (gamma_gas-1)/((gamma_gas+1)*nozzle_efficiency))**(-gamma_gas/(gamma_gas-1)) # choked_ration = P50/at choked_pressure 
			choked_pressure = P50/choked_ration
			# =============================
			# Nozzle is not chocked
			if atm_pressure >= choked_pressure:
				isp_ratio = (atm_pressure/P50)**((gamma_gas-1)/gamma_gas)
				T90 = (1- (1- isp_ratio)*nozzle_efficiency)*T50 # Exhaust Gas Temperature  # nozzle_efficiency = (turbine exit temperature -Nozzle exit temperature) / (turbine exit temperature-ideal Nozzle exit temperature)
				exit_velocity = sqrt(2*cp_gas*(T50-T90))
			#==============================
			# Nozzle is chocked case
			else:
				isp_ratio = (choked_pressure/P50)**((gamma_gas-1)/gamma_gas) # nozzle_efficiency = (turbine exit temperature -Nozzle exit temperature) / (turbine exit temperature-ideal Nozzle exit temperature)
				T90 = (1- (1- isp_ratio)*nozzle_efficiency)*T50  # nozzle_efficiency = (turbine exit temperature -Nozzle exit temperature) / (turbine exit temperature-ideal Nozzle exit temperature)
				exit_velocity = sqrt(gamma_gas*R*T90)
			return T90, exit_velocity
		machine_df[['T90','Ve']] =  machine_df.apply(lambda row: EGT(row['P50'],row['T50'],row['Atm_pressure']),axis = 1,result_type="expand") 
		machine_df['EGT_margin'] =  machine_df['T90'].apply(lambda x: 1223.15-x)
		machine_df['Nc/Nf']      =  machine_df['Nc']/machine_df['Nf']
		machine_df['PCNcRdmd '] = (machine_df.Nc-machine_df.NRf)/machine_df.Nc
		#######################################
		# Mass of cold air fan outlet
		def Mass_flow (phi,Ps30,fuel_air_ratio,BPR):
			Mass_flow_fuel = phi*Ps30*0.0001259978805556
			Mass_flow_hotair = Mass_flow_fuel/fuel_air_ratio 
			Mass_flow_coldair = BPR*Mass_flow_hotair 
			return Mass_flow_hotair,Mass_flow_coldair 
		machine_df[['M_Hot','M_cold']] = machine_df.apply(lambda row: Mass_flow(row['phi'],row['Ps30'],row['fuel_air_ratio'],row['BPR']),axis=1,result_type="expand")
	    ############################################
		def fan_exit_temperature(T2,T24,T48,T50,fuel_air_ratio,BPR):
			T21 = T2+ (((1+fuel_air_ratio)*(T48-T50)*(cp_gas/cp_air)*transmission_efficiency - (T24-T2))/BPR)
			return T21
		machine_df['T21'] = machine_df.apply(lambda row: fan_exit_temperature(row['T2'],row['T24'],row['T48'],row['T50'],row['fuel_air_ratio'],row['BPR']),axis=1) 
		###########################################
		def Fan_pressure_ratio(Atm_pressure,P2, T21,T2):
			ratio= (1+fan_efficiency*((T21/T2)-1))**(gamma_air/(gamma_air-1))
			P21 = ratio*P2
			choked_ration =  (1- (gamma_air-1)/((gamma_air+1)*nozzle_efficiency))**(-gamma_air/(gamma_air-1)) # choked_ration = P21/at choked_pressure  
			choked_pressure = P21/choked_ration
			# =============================
			# Nozzle is not chocked 
			if Atm_pressure	 >= choked_pressure:
				isp_ratio = (Atm_pressure	/P21)**((gamma_air-1)/gamma_air)  # we need to consider atm_pressure
				exit_temp_nozzle = (1- (1- isp_ratio)*nozzle_efficiency)*T21 # Exhaust Gas Temperature  # nozzle_efficiency = (bypass nozzle exit temperature -Nozzle exit temperature) / (fan exit temperature-ideal Nozzle exit temperature)
				exit_velocity = sqrt(2*cp_air*(T21-exit_temp_nozzle))
			#==============================
	    	# Nozzle is chocked case 
			else:
				isp_ratio = (choked_pressure/P21)**((gamma_air-1)/gamma_air) # nozzle_efficiency = (bypass nozzle exit temperature -Nozzle exit temperature) / (fan exit temperature-ideal Nozzle exit temperature)
				exit_temp_nozzle = (1- (1- isp_ratio)*nozzle_efficiency)*T21  # nozzle_efficiency = (turbine exit temperature -Nozzle exit temperature) / (turbine exit temperature-ideal Nozzle exit temperature)
				exit_velocity = sqrt(gamma_air*R*exit_temp_nozzle) 
			return P21,exit_velocity
		machine_df[['P21','V_bypass']] =  machine_df.apply(lambda row:  Fan_pressure_ratio(row['Atm_pressure'],row['P2'],row['T21'],row['T2']),axis=1,result_type="expand")
		def Corrected_flow(M_cold,M_Hot,T2,P2,T21,P21):
			atm_temperature =    228.15
			atm_pressure    =    1.01325
			w_f = (M_cold*sqrt(T2/atm_temperature))/(P2/atm_pressure) #fan_corrected_mass_flow 
			return w_f
		machine_df['W_f'] = machine_df.apply(lambda row: Corrected_flow(row['M_cold'],row['M_Hot'],row['T2'],row['P2'],row['T21'],row['P21']),axis=1,result_type="expand")
	    ###############################################################
	    # Thrust for a turbofan engine
		def thrust(fuel_air_ratio,BPR,M_Hot,V_bypass,Ve,Va):
			Fan_thrust = BPR*M_Hot*V_bypass - BPR*M_Hot*Va
			core_thrust = M_Hot*(1+fuel_air_ratio)*Ve - M_Hot*Va 
			Thrust= M_Hot*(1+fuel_air_ratio)*Ve + BPR*M_Hot*V_bypass - (1+BPR)*M_Hot*Va 
			TSFC = (fuel_air_ratio*M_Hot*10**6)/Thrust # TSFC may also be thought of as fuel consumption (grams/second) per unit of thrust (kilonewtons, or kN).
			return Thrust,Fan_thrust,core_thrust,TSFC
		machine_df[['Thrust','Fan_thrust','core_thrust','TSFC']] = machine_df.apply(lambda row: thrust(row['fuel_air_ratio'],row['BPR'],row['M_Hot'],row['V_bypass'],row['Ve'],row['Va']),axis = 1,result_type="expand")
	    ############################################################
		def Thermal_efficiency(fuel_air_ratio,BPR,V_bypass,Ve,Va,Thrust,M_Hot,M_cold):
		    Q_R = 42000*1000 # Heat of reaction of the fuel
		    Thermal_efficiency = (((1+fuel_air_ratio)*(Ve)**2 + BPR*(V_bypass)**2 - (1+BPR)*(Va)**2)/(2*fuel_air_ratio*Q_R))*100
		    return Thermal_efficiency 
		machine_df['Thermal_efficiency']= machine_df.apply(lambda row: Thermal_efficiency(row['fuel_air_ratio'],row['BPR'],row['V_bypass'],row['Ve'],row['Va'],row['Thrust'],row['M_Hot'],row['M_cold']),axis = 1) 
		#############################################################
		machine_df = machine_df.loc[:,imp_col]
		return machine_df
	@st.cache 
	def scaling(df,path_sc,machine_id):
		mfile = BytesIO(requests.get(path_sc).content)
		scaler =   pickle.load(mfile)
		cols = df.columns.difference(['Engine_No']) 
		
		scaler.clip = False #added based on stackoverflow suggestion

		norm_train_df = pd.DataFrame(scaler.transform(df[cols]), columns=cols, index=df.index)
		join_df = df[df.columns.difference(cols)].join(norm_train_df) 
		sc_df= join_df.reindex(columns = df.columns) 
		machine_df= sc_df[sc_df['Engine_No'] == machine_id].copy().reset_index(drop=True).drop(['Engine_No'], axis = 1)
		return machine_df
	@st.cache(allow_output_mutation=True)
	def load_ML_model(model_url):
		ml_load = BytesIO(requests.get(model_url).content)
		loaded_model =   pickle.load(ml_load)
		return loaded_model 

	if data_set_type == "FD001":
		st.write("Selected raw data file from")
		data_url = 'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/test_FD001.txt' 
		st.markdown(data_url, unsafe_allow_html=True) 
		df = imp_features(data_url) 
		sc_url = r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD001/scaler.pkl?raw=true"
		model_url  = r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD001/logistic_model_cali.pkl?raw=true"
		ref_model_url= r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD001/RF_model_RUL.pkl?raw=true"
	elif data_set_type == "FD002":
		st.write("Selected raw data file from")
		data_url = 'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/test_FD002.txt' 
		st.markdown(data_url, unsafe_allow_html=True) 
		df = imp_features(data_url)
		sc_url = r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD002/scaler.pkl?raw=true"
		model_url = r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD002/logistic_model_cali.pkl?raw=true"
		ref_model_url= r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD002/RF_model_RUL.pkl?raw=true"
	elif data_set_type == "FD003":
		st.write("Selected raw data file from")
		data_url = 'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/test_FD003.txt' 
		st.markdown(data_url, unsafe_allow_html=True) 
		df = imp_features(data_url)
		sc_url = r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD003/scaler.pkl?raw=true"
		model_url = r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD003/logistic_model_cali.pkl?raw=true"
		ref_model_url= r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD003/RF_model_RUL.pkl?raw=true"
	else:
		st.write("Selected raw data file from")
		data_url = 'https://raw.githubusercontent.com/gethgle/predictive-maintenance/main/data/test_FD004.txt' 
		st.markdown(data_url, unsafe_allow_html=True) 
		df = imp_features(data_url)
		sc_url = r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD004/scaler.pkl?raw=true"
		model_url = r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD004/logistic_model_cali.pkl?raw=true"
		ref_model_url= r"https://github.com/gethgle/predictive-maintenance/blob/main/data/output/FD004/RF_model_RUL.pkl?raw=true"

	#pro_type = st.selectbox("Do you want predict probability of failure (OR) How many life cycle are still remaining", ("Classification", "Regression"))
	engine_id = st.sidebar.selectbox("Which engine do you want to predict for?", df.Engine_No.unique())
	st.sidebar.write("Select any engine from",df.Engine_No.unique()[0],"to",df.Engine_No.unique()[-1]) 
	time_of_cycle = st.sidebar.select_slider("At a given point of time, check whether the engine will fail or not",list(range(1, df[df['Engine_No']==engine_id].shape[0]))) 
	st.sidebar.write("The Engine has  completed ",time_of_cycle," cycles and we are predicting what happened in next 30 cycle")


	scal = scaling(df,sc_url,engine_id)
	model = load_ML_model(model_url)
	test_point = scal.loc[[time_of_cycle],:]
	#yhat = model.predict_proba(test_point) #commented as it was throwing errors 'callibrator' attribute

	model_reg = load_ML_model(ref_model_url)
	y_pred_test = model_reg.predict(test_point)
	
	#col1, col2 = st.beta_columns(2) 
	col1, col2 = st.columns(2) 
	
	text = "**Probability of Engine " +str(engine_id)+ " failure within next 30 life cycle:** (hard coded for now)"
	col1.write(text)
	# Pie chart
	labels = ['works', 'Fails']
	#sizes =  yhat.tolist()
	sizes = [0.30, 0.70] #hard coded
	
	#colors
	colors = ['#ff9999', '#c2c2f0']
	fig1, ax1 = plt.subplots()
	ax1.pie(sizes, colors = colors, labels=labels, pctdistance=0.35,autopct='%1.1f%%', startangle=90)
	#draw circle
	centre_circle = plt.Circle((0,0),0.6,fc='white')
	fig = plt.gcf()
	fig.gca().add_artist(centre_circle)
	# Equal aspect ratio ensures that pie is drawn as a circle
	ax1.axis('equal')  
	plt.tight_layout()
	col1.pyplot()
	text_1 = "**Remaining useful life of Engine " + str(engine_id) + ":**"
	
	col2.write(text_1)

	col2.markdown("----")

	text_2 = "**Predicted cycles remaining:** " +str(math.ceil(y_pred_test[0])) 
	col2.write(text_2) 


     


#create a button in the side bar that will move to the next page/radio button choice
st.sidebar.title("Navigation") 
app_mode = st.sidebar.selectbox("Go to", ["Home","Data Exploration","Feature Engineering","Model Building","Model prediction"]) 
if app_mode == 'Home':
	load_homepage()
elif app_mode== "Data Exploration":
	EDA()
elif app_mode == "Feature Engineering":
	FE()
elif app_mode == "Model Building":
	MS()
else:
	model_predi()

