import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

class TennisPredictions:

	def __init__(self, input_files_path, test_file):
		self.input_files_path = input_files_path;
		self.input_data = None;
		self.test_file = test_file;

	# Function to get data
	def load_data(self):
		print 'Reading data:'
		list_of_files = os.listdir(self.input_files_path)
		print str(len(list_of_files)) + 'Files found'
		input_data_initialized = False;
		for file_name in list_of_files:
			if '.csv' in file_name:
				print 'processing file ' + file_name;
				#if input_data is not initialized
				if not input_data_initialized:
					data = pd.read_csv(self.input_files_path + file_name);
					self.player_names = data['Player1'].copy()
					self.player_names = self.player_names.append(data['Player2']);
					if file_name != self.test_file:
						self.input_data = pd.read_csv(self.input_files_path + file_name);
						#print 'Data shape: ' + str(self.input_data.shape)
						input_data_initialized = True;
				else:
					data = pd.read_csv(self.input_files_path + file_name);
					self.player_names = self.player_names.append(data['Player1']);
					self.player_names = self.player_names.append(data['Player2']);
					if file_name != self.test_file:
						#print 'Data shape: ' + str(data.shape)
						self.input_data = self.input_data.append(data);
		print 'Input data shape: ' +  str(self.input_data.shape)
		
	def clean_data(self, columns_to_clean = []):
		print 'Cleaning data...'
		self.columns_to_clean = columns_to_clean;
		#Lets convert the player names into numeric classes first
		label_encoder = preprocessing.LabelEncoder()
		self.names_fit = label_encoder.fit(self.player_names);
		self.input_data['Player1'] = self.names_fit.transform(self.input_data['Player1']);
		self.input_data['Player2'] = self.names_fit.transform(self.input_data['Player2']);
		#print str(self.input_data['Player2'])
		#imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
		#self.input_data[columns_to_clean] = imp.fit_transform(self.input_data[columns_to_clean])
		self.input_data[self.input_data.isnull()] = 0;

	def apply_logstic_regression(self, data_columns, target):
		self.data_columns = data_columns;
		self.target_column = target;
		self.data_y = self.input_data[target];
		self.data_x = self.input_data[data_columns];
		#print str(self.input_data[:5])
		self.clf = linear_model.LogisticRegression();
		self.clf.fit(self.data_x, self.data_y);
		
	def apply_linear_regression(self, data_columns, target):
		self.data_columns = data_columns;
		self.target_column = target;
		self.data_y = self.input_data[target];
		self.data_x = self.input_data[data_columns];
		self.clf = linear_model.LinearRegression();
		self.clf.fit(self.data_x, self.data_y);
	
	"""def test_predictions(self):
		self.test_data = pd.read_csv(self.test_file);
		imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
		self.test_data[self.columns_to_clean] = imp.fit_transform(self.test_data[self.columns_to_clean])
		self.predicted_putput = self.clf.predict(self.test_data[self.data_columns]);
		#print str(self.predicted_putput);
		total_match = 0;
		total_mismatch = 0;
		i = 0;
		for act_result in self.test_data[self.target_column]:
			predicted_result = self.predicted_putput[i];
			if predicted_result == act_result:
				total_match += 1;
			else:
				total_mismatch += 1; 
			i += 1;
		print "Test result on file(" + self.test_file + ")"
		print "\tNumber of tests: " + str(total_match + total_mismatch);
		print "\t\tCorrect results: " + str(total_match)
		print "\t\tIncorrect results: " + str(total_mismatch)"""
	
	def test_predictions(self, allowed_error, is_linear = True):
		self.test_data = pd.read_csv(self.input_files_path + self.test_file);
		self.test_data['Player1'] = self.names_fit.transform(self.test_data['Player1']);
		self.test_data['Player2'] = self.names_fit.transform(self.test_data['Player2']);
		#imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
		#self.test_data[self.columns_to_clean] = imp.fit_transform(self.test_data[self.columns_to_clean])
		self.test_data[self.test_data.isnull()] = 0
		self.predicted_putput = self.clf.predict(self.test_data[self.data_columns]);
		#print str(self.predicted_putput);
		total_match = 0;
		total_mismatch = 0;
		i = 0;
		print "Test result on file(" + self.test_file + ") on data: " + self.target_column;
		for act_result in self.test_data[self.target_column]:
			if is_linear:
				predicted_result = math.ceil(self.predicted_putput[i]);
				try:
					error = round(abs((predicted_result - act_result)*100/act_result), 2);
				except:
					error = 'Inf'
				print "Act: " + str(act_result) + ', Predicted: ' + str(predicted_result) + ', Error: ' + str(error);
				if error <= allowed_error:
					total_match += 1;
				else:
					total_mismatch += 1; 
				i += 1;
			else:
				predicted_result = self.predicted_putput[i];
				error = int(predicted_result) - int(act_result);
				print "Act: " + str(act_result) + ', Predicted: ' + str(predicted_result) + ', Error: ' + str(error);
				if error == 0:
					total_match += 1;
				else:
					total_mismatch += 1; 
				i += 1;
		print "\tNumber of tests: " + str(total_match + total_mismatch);
		print "\t\tCorrect results: " + str(total_match)
		print "\t\tIncorrect results: " + str(total_mismatch)

	def predict_match(self, data):
		outcome = self.clf.predict(data);
		return outcome[0];

if __name__ == '__main__':
	#'Wimbledon-men-2013.csv'
	#'AusOpen-men-2013.csv'
	tennis_predictions = TennisPredictions('data/', 'Wimbledon-men-2013.csv');
	tennis_predictions.load_data();
	total_data_columns = ['Player1', 'Player2', 'Result', 'Round', 'FNL.1', 'FNL.2', 'FSP.1', 'FSW.1', 'SSP.1', 'SSW.1', 'ACE.1', 'DBF.1', 'WNR.1', 'UFE.1', 'BPC.1', 'BPW.1', 'NPA.1', 'NPW.1', 'TPW.1', 'ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1', 'FSP.2', 'FSW.2', 'SSP.2', 'SSW.2', 'ACE.2', 'DBF.2', 'WNR.2', 'UFE.2', 'BPC.2', 'BPW.2', 'NPA.2', 'NPW.2', 'TPW.2', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2'];
	tennis_predictions.clean_data(total_data_columns);
	
	data_columns = ['Player1', 'Player2', 'Round', 'FNL.1', 'FNL.2', 'FSP.1', 'FSW.1', 'SSP.1', 'SSW.1', 'ACE.1', 'DBF.1', 'WNR.1', 'UFE.1', 'BPC.1', 'BPW.1', 'NPA.1', 'NPW.1', 'TPW.1', 'ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1', 'FSP.2', 'FSW.2', 'SSP.2', 'SSW.2', 'ACE.2', 'DBF.2', 'WNR.2', 'UFE.2', 'BPC.2', 'BPW.2', 'NPA.2', 'NPW.2', 'TPW.2', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2'];
	tennis_predictions.apply_logstic_regression(data_columns, 'Result');
	tennis_predictions.test_predictions(2, False);
	
	
	data_columns = ['Player1', 'Player2', 'Result', 'Round', 'FNL.1', 'FNL.2', 'FSP.1', 'FSW.1', 'SSP.1', 'SSW.1', 'DBF.1', 'WNR.1', 'UFE.1', 'BPC.1', 'BPW.1', 'NPA.1', 'NPW.1', 'TPW.1', 'ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1', 'FSP.2', 'FSW.2', 'SSP.2', 'SSW.2', 'ACE.2', 'DBF.2', 'WNR.2', 'UFE.2', 'BPC.2', 'BPW.2', 'NPA.2', 'NPW.2', 'TPW.2', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2'];
	tennis_predictions.apply_linear_regression(data_columns, 'ACE.1');
	tennis_predictions.test_predictions(2);


	data_columns = ['Player1', 'Player2', 'Result', 'Round', 'FNL.1', 'FNL.2', 'FSP.1', 'FSW.1', 'SSP.1', 'SSW.1', 'ACE.1', 'WNR.1', 'UFE.1', 'BPC.1', 'BPW.1', 'NPA.1', 'NPW.1', 'TPW.1', 'ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1', 'FSP.2', 'FSW.2', 'SSP.2', 'SSW.2', 'ACE.2', 'DBF.2', 'WNR.2', 'UFE.2', 'BPC.2', 'BPW.2', 'NPA.2', 'NPW.2', 'TPW.2', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2'];
	tennis_predictions.apply_linear_regression(data_columns, 'DBF.1');
	tennis_predictions.test_predictions(2);
