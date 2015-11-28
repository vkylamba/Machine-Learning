
# This procedure does some machine learning tasks on the UCI Tennis data. The data is stored in the form of CSV files in current folder. We will use python to implement machine learning.


```python
#Lets load os module to fine the files present in the directory
import os
```

Our csv files are in the data folder. So let's see the files first


```python
data_files_path = 'data/'
data_files = os.listdir(data_files_path)
```


```python
print data_files
```

    ['Wimbledon-women-2013.csv', 'FrenchOpen-women-2013.csv', 'AusOpen-men-2013.csv', 'FrenchOpen-men-2013.csv', 'Wimbledon-men-2013.csv', 'AusOpen-women-2013.csv', 'USOpen-women-2013.csv', 'USOpen-men-2013.csv']


Now lets load the data from the files. We will use pandas module to load the data.


```python
import pandas as pd
input_data = pd.read_csv(data_files_path + data_files[0]);
```

Now lets see how does the data looks. Lets see first 5 rows of the data.


```python
input_data[:5]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player1</th>
      <th>Player2</th>
      <th>Round</th>
      <th>Result</th>
      <th>FNL.1</th>
      <th>FNL.2</th>
      <th>FSP.1</th>
      <th>FSW.1</th>
      <th>SSP.1</th>
      <th>SSW.1</th>
      <th>...</th>
      <th>BPC.2</th>
      <th>BPW.2</th>
      <th>NPA.2</th>
      <th>NPW.2</th>
      <th>TPW.2</th>
      <th>ST1.2</th>
      <th>ST2.2</th>
      <th>ST3.2</th>
      <th>ST4.2</th>
      <th>ST5.2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>    M.Koehler</td>
      <td> V.Azarenka</td>
      <td> 1</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 60</td>
      <td> 21</td>
      <td> 40</td>
      <td>  8</td>
      <td>...</td>
      <td> 16</td>
      <td> 6</td>
      <td>  8</td>
      <td>  4</td>
      <td>NaN</td>
      <td> 6</td>
      <td> 6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>   E.Baltacha</td>
      <td> F.Pennetta</td>
      <td> 1</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 69</td>
      <td> 23</td>
      <td> 31</td>
      <td>  6</td>
      <td>...</td>
      <td>  6</td>
      <td> 5</td>
      <td> 14</td>
      <td> 11</td>
      <td>NaN</td>
      <td> 6</td>
      <td> 6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>    S-W.Hsieh</td>
      <td>    T.Maria</td>
      <td> 1</td>
      <td> 1</td>
      <td> 2</td>
      <td> 0</td>
      <td> 63</td>
      <td> 17</td>
      <td> 37</td>
      <td> 10</td>
      <td>...</td>
      <td>  1</td>
      <td> 0</td>
      <td>  8</td>
      <td>  2</td>
      <td>NaN</td>
      <td> 1</td>
      <td> 0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>     A.Cornet</td>
      <td>     V.King</td>
      <td> 1</td>
      <td> 1</td>
      <td> 2</td>
      <td> 1</td>
      <td> 57</td>
      <td> 36</td>
      <td> 43</td>
      <td> 21</td>
      <td>...</td>
      <td>  4</td>
      <td> 1</td>
      <td> 48</td>
      <td> 32</td>
      <td>NaN</td>
      <td> 6</td>
      <td> 3</td>
      <td>  1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td> Y.Putintseva</td>
      <td> K.Flipkens</td>
      <td> 1</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 73</td>
      <td> 34</td>
      <td> 27</td>
      <td> 12</td>
      <td>...</td>
      <td>  9</td>
      <td> 3</td>
      <td> 35</td>
      <td> 24</td>
      <td>NaN</td>
      <td> 7</td>
      <td> 6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>



Lets see the Player1 column for first 5 rows


```python
input_data['Player1'][:5]
```




    0       M.Koehler
    1      E.Baltacha
    2       S-W.Hsieh
    3        A.Cornet
    4    Y.Putintseva
    Name: Player1, dtype: object



Now lets make a function to load data from all the data files. Since each file belongs to certain type of match/tournament. We will add additional column indiacting the file name or tournament type.


```python
def load_data(data_files):
    input_data = None;
    input_data_initialized = False; # to check the state of input_data variable
    for file_name in data_files:
        print 'Reading from file ' + file_name
        if not input_data_initialized:
            #Initialize the input_data
            data = pd.read_csv(data_files_path + file_name);
            input_data = data;
            input_data['Type'] = file_name.split('.')[0];
            input_data_initialized = True;
        else:
            #Store the data into data variable
            data = pd.read_csv(data_files_path + file_name);
            data['Type'] = file_name.split('.')[0];
            #Append the data into input_data
            input_data = input_data.append(data);
    return input_data;
```


```python
#Lets load data from all the files
input_data = load_data(data_files);
```

    Reading from file Wimbledon-women-2013.csv
    Reading from file FrenchOpen-women-2013.csv
    Reading from file AusOpen-men-2013.csv
    Reading from file FrenchOpen-men-2013.csv
    Reading from file Wimbledon-men-2013.csv
    Reading from file AusOpen-women-2013.csv
    Reading from file USOpen-women-2013.csv
    Reading from file USOpen-men-2013.csv


Lets check the size of data


```python
input_data.shape
```




    (943, 43)



This means the input_data have 943 rows and 42 columns

Lets see first 5 rows of input_data


```python
input_data[:5]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player1</th>
      <th>Player2</th>
      <th>Round</th>
      <th>Result</th>
      <th>FNL.1</th>
      <th>FNL.2</th>
      <th>FSP.1</th>
      <th>FSW.1</th>
      <th>SSP.1</th>
      <th>SSW.1</th>
      <th>...</th>
      <th>BPW.2</th>
      <th>NPA.2</th>
      <th>NPW.2</th>
      <th>TPW.2</th>
      <th>ST1.2</th>
      <th>ST2.2</th>
      <th>ST3.2</th>
      <th>ST4.2</th>
      <th>ST5.2</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>    M.Koehler</td>
      <td> V.Azarenka</td>
      <td> 1</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 60</td>
      <td> 21</td>
      <td> 40</td>
      <td>  8</td>
      <td>...</td>
      <td> 6</td>
      <td>  8</td>
      <td>  4</td>
      <td>NaN</td>
      <td> 6</td>
      <td> 6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> Wimbledon-women-2013</td>
    </tr>
    <tr>
      <th>1</th>
      <td>   E.Baltacha</td>
      <td> F.Pennetta</td>
      <td> 1</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 69</td>
      <td> 23</td>
      <td> 31</td>
      <td>  6</td>
      <td>...</td>
      <td> 5</td>
      <td> 14</td>
      <td> 11</td>
      <td>NaN</td>
      <td> 6</td>
      <td> 6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> Wimbledon-women-2013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>    S-W.Hsieh</td>
      <td>    T.Maria</td>
      <td> 1</td>
      <td> 1</td>
      <td> 2</td>
      <td> 0</td>
      <td> 63</td>
      <td> 17</td>
      <td> 37</td>
      <td> 10</td>
      <td>...</td>
      <td> 0</td>
      <td>  8</td>
      <td>  2</td>
      <td>NaN</td>
      <td> 1</td>
      <td> 0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> Wimbledon-women-2013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>     A.Cornet</td>
      <td>     V.King</td>
      <td> 1</td>
      <td> 1</td>
      <td> 2</td>
      <td> 1</td>
      <td> 57</td>
      <td> 36</td>
      <td> 43</td>
      <td> 21</td>
      <td>...</td>
      <td> 1</td>
      <td> 48</td>
      <td> 32</td>
      <td>NaN</td>
      <td> 6</td>
      <td> 3</td>
      <td>  1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> Wimbledon-women-2013</td>
    </tr>
    <tr>
      <th>4</th>
      <td> Y.Putintseva</td>
      <td> K.Flipkens</td>
      <td> 1</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 73</td>
      <td> 34</td>
      <td> 27</td>
      <td> 12</td>
      <td>...</td>
      <td> 3</td>
      <td> 35</td>
      <td> 24</td>
      <td>NaN</td>
      <td> 7</td>
      <td> 6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> Wimbledon-women-2013</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>



Lets read all the player names and match types


```python
player_names = input_data['Player1'].copy();
player_names = player_names.append(input_data['Player2']);

tournament_types = input_data['Type'].copy();
```


```python
player_names[:5]
```




    0       M.Koehler
    1      E.Baltacha
    2       S-W.Hsieh
    3        A.Cornet
    4    Y.Putintseva
    dtype: object



We need to divide the data into two parts. One to train the learning algorithms, and other part to test the algorithms. We will select 100 rows randomaly for test data.


```python
import numpy as np
#Lets randomize the rows in input_data
input_data = input_data.take(np.random.permutation(len(input_data))[:])
#Lets take first 100 rows now
test_data = input_data[:100].copy();
input_data = input_data[100:-1];
```

Test data looks like:


```python
test_data.shape
```




    (100, 43)




```python
input_data.shape
```




    (842, 43)



The Result column have results of all the matches. 
The result is 1 if Player1 won the match, it is zero if Player2 won the match. We will be feeding all this data to learning algorithm. The algorithm accepts numeric values only. So we need to convert the player names and match types into numeric values. We will be identifying each player with a unique number. To do so lets import preprocessing module of sklearn, and use the label encoder availabel in it.


```python
from sklearn import preprocessing
name_encoder = preprocessing.LabelEncoder()
tournament_encoder = preprocessing.LabelEncoder()
```

Now lets transform the names into numeric values


```python
names_fit = name_encoder.fit(player_names)
tournament_fit = tournament_encoder.fit(tournament_types)
input_data['Player1'] = names_fit.transform(input_data['Player1']);
input_data['Player2'] = names_fit.transform(input_data['Player2']);
input_data['Type'] = tournament_fit.transform(input_data['Type']);

test_data['Player1'] = names_fit.transform(test_data['Player1']);
test_data['Player2'] = names_fit.transform(test_data['Player2']);
test_data['Type'] = tournament_fit.transform(test_data['Type']);
```


```python
input_data[['Player1', 'Player2', 'Type']][:5]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player1</th>
      <th>Player2</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td> 244</td>
      <td> 464</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>18</th>
      <td> 164</td>
      <td> 367</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>59</th>
      <td> 535</td>
      <td> 650</td>
      <td> 5</td>
    </tr>
    <tr>
      <th>57</th>
      <td> 470</td>
      <td>  37</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>56</th>
      <td> 429</td>
      <td> 331</td>
      <td> 0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data[['Player1', 'Player2', 'Type']][:5]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player1</th>
      <th>Player2</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84</th>
      <td> 124</td>
      <td> 218</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>  43</td>
      <td> 562</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>80</th>
      <td> 100</td>
      <td> 525</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>86</th>
      <td> 439</td>
      <td> 198</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>7 </th>
      <td>   8</td>
      <td> 134</td>
      <td> 5</td>
    </tr>
  </tbody>
</table>
</div>



And the input_data looks like:


```python
input_data[:5]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player1</th>
      <th>Player2</th>
      <th>Round</th>
      <th>Result</th>
      <th>FNL.1</th>
      <th>FNL.2</th>
      <th>FSP.1</th>
      <th>FSW.1</th>
      <th>SSP.1</th>
      <th>SSW.1</th>
      <th>...</th>
      <th>BPW.2</th>
      <th>NPA.2</th>
      <th>NPW.2</th>
      <th>TPW.2</th>
      <th>ST1.2</th>
      <th>ST2.2</th>
      <th>ST3.2</th>
      <th>ST4.2</th>
      <th>ST5.2</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td> 244</td>
      <td> 464</td>
      <td> 1</td>
      <td> 1</td>
      <td> 3</td>
      <td> 0</td>
      <td> 66</td>
      <td> 45</td>
      <td> 34</td>
      <td> 14</td>
      <td>...</td>
      <td>  6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>  67</td>
      <td> 1</td>
      <td> 4</td>
      <td>  4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>18</th>
      <td> 164</td>
      <td> 367</td>
      <td> 1</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 47</td>
      <td> 17</td>
      <td> 53</td>
      <td>  9</td>
      <td>...</td>
      <td>  7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>  58</td>
      <td> 6</td>
      <td> 6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>59</th>
      <td> 535</td>
      <td> 650</td>
      <td> 1</td>
      <td> 1</td>
      <td> 2</td>
      <td> 0</td>
      <td> 54</td>
      <td> 18</td>
      <td> 46</td>
      <td> 16</td>
      <td>...</td>
      <td>  3</td>
      <td> 11</td>
      <td>  7</td>
      <td> NaN</td>
      <td> 3</td>
      <td> 4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> 5</td>
    </tr>
    <tr>
      <th>57</th>
      <td> 470</td>
      <td>  37</td>
      <td> 1</td>
      <td> 1</td>
      <td> 2</td>
      <td> 0</td>
      <td> 69</td>
      <td> 24</td>
      <td> 31</td>
      <td>  4</td>
      <td>...</td>
      <td>  4</td>
      <td>  9</td>
      <td>  6</td>
      <td> NaN</td>
      <td> 4</td>
      <td> 2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>56</th>
      <td> 429</td>
      <td> 331</td>
      <td> 1</td>
      <td> 0</td>
      <td> 2</td>
      <td> 3</td>
      <td> 61</td>
      <td> 69</td>
      <td> 39</td>
      <td> 29</td>
      <td>...</td>
      <td> 15</td>
      <td> 11</td>
      <td> 16</td>
      <td> 167</td>
      <td> 6</td>
      <td> 5</td>
      <td>  6</td>
      <td>  4</td>
      <td>  6</td>
      <td> 0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>



Now we see that some data points are 'NaN', which is not a numeric value. So we need to replace it with numeric values. We will replace NaN values with 0. To check which values are 'NaN' there is isnul function which returns True if the value is null.


```python
input_data.isnull()[:5]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player1</th>
      <th>Player2</th>
      <th>Round</th>
      <th>Result</th>
      <th>FNL.1</th>
      <th>FNL.2</th>
      <th>FSP.1</th>
      <th>FSW.1</th>
      <th>SSP.1</th>
      <th>SSW.1</th>
      <th>...</th>
      <th>BPW.2</th>
      <th>NPA.2</th>
      <th>NPW.2</th>
      <th>TPW.2</th>
      <th>ST1.2</th>
      <th>ST2.2</th>
      <th>ST3.2</th>
      <th>ST4.2</th>
      <th>ST5.2</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td>...</td>
      <td> False</td>
      <td>  True</td>
      <td>  True</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td>  True</td>
      <td>  True</td>
      <td> False</td>
    </tr>
    <tr>
      <th>18</th>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td>...</td>
      <td> False</td>
      <td>  True</td>
      <td>  True</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td>  True</td>
      <td>  True</td>
      <td>  True</td>
      <td> False</td>
    </tr>
    <tr>
      <th>59</th>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td>...</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td>  True</td>
      <td> False</td>
      <td> False</td>
      <td>  True</td>
      <td>  True</td>
      <td>  True</td>
      <td> False</td>
    </tr>
    <tr>
      <th>57</th>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td>...</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td>  True</td>
      <td> False</td>
      <td> False</td>
      <td>  True</td>
      <td>  True</td>
      <td>  True</td>
      <td> False</td>
    </tr>
    <tr>
      <th>56</th>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td>...</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
      <td> False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>



Lets replace all the null values with 0.


```python
input_data[input_data.isnull()] = 0
test_data[test_data.isnull()] = 0
```


```python
input_data[:5]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player1</th>
      <th>Player2</th>
      <th>Round</th>
      <th>Result</th>
      <th>FNL.1</th>
      <th>FNL.2</th>
      <th>FSP.1</th>
      <th>FSW.1</th>
      <th>SSP.1</th>
      <th>SSW.1</th>
      <th>...</th>
      <th>BPW.2</th>
      <th>NPA.2</th>
      <th>NPW.2</th>
      <th>TPW.2</th>
      <th>ST1.2</th>
      <th>ST2.2</th>
      <th>ST3.2</th>
      <th>ST4.2</th>
      <th>ST5.2</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td> 244</td>
      <td> 464</td>
      <td> 1</td>
      <td> 1</td>
      <td> 3</td>
      <td> 0</td>
      <td> 66</td>
      <td> 45</td>
      <td> 34</td>
      <td> 14</td>
      <td>...</td>
      <td>  6</td>
      <td>  0</td>
      <td>  0</td>
      <td>  67</td>
      <td> 1</td>
      <td> 4</td>
      <td> 4</td>
      <td> 0</td>
      <td> 0</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>18</th>
      <td> 164</td>
      <td> 367</td>
      <td> 1</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 47</td>
      <td> 17</td>
      <td> 53</td>
      <td>  9</td>
      <td>...</td>
      <td>  7</td>
      <td>  0</td>
      <td>  0</td>
      <td>  58</td>
      <td> 6</td>
      <td> 6</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>59</th>
      <td> 535</td>
      <td> 650</td>
      <td> 1</td>
      <td> 1</td>
      <td> 2</td>
      <td> 0</td>
      <td> 54</td>
      <td> 18</td>
      <td> 46</td>
      <td> 16</td>
      <td>...</td>
      <td>  3</td>
      <td> 11</td>
      <td>  7</td>
      <td>   0</td>
      <td> 3</td>
      <td> 4</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 5</td>
    </tr>
    <tr>
      <th>57</th>
      <td> 470</td>
      <td>  37</td>
      <td> 1</td>
      <td> 1</td>
      <td> 2</td>
      <td> 0</td>
      <td> 69</td>
      <td> 24</td>
      <td> 31</td>
      <td>  4</td>
      <td>...</td>
      <td>  4</td>
      <td>  9</td>
      <td>  6</td>
      <td>   0</td>
      <td> 4</td>
      <td> 2</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>56</th>
      <td> 429</td>
      <td> 331</td>
      <td> 1</td>
      <td> 0</td>
      <td> 2</td>
      <td> 3</td>
      <td> 61</td>
      <td> 69</td>
      <td> 39</td>
      <td> 29</td>
      <td>...</td>
      <td> 15</td>
      <td> 11</td>
      <td> 16</td>
      <td> 167</td>
      <td> 6</td>
      <td> 5</td>
      <td> 6</td>
      <td> 4</td>
      <td> 6</td>
      <td> 0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>




```python
test_data[:5]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player1</th>
      <th>Player2</th>
      <th>Round</th>
      <th>Result</th>
      <th>FNL.1</th>
      <th>FNL.2</th>
      <th>FSP.1</th>
      <th>FSW.1</th>
      <th>SSP.1</th>
      <th>SSW.1</th>
      <th>...</th>
      <th>BPW.2</th>
      <th>NPA.2</th>
      <th>NPW.2</th>
      <th>TPW.2</th>
      <th>ST1.2</th>
      <th>ST2.2</th>
      <th>ST3.2</th>
      <th>ST4.2</th>
      <th>ST5.2</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84</th>
      <td> 124</td>
      <td> 218</td>
      <td> 2</td>
      <td> 1</td>
      <td> 2</td>
      <td> 1</td>
      <td> 73</td>
      <td> 57</td>
      <td> 27</td>
      <td> 17</td>
      <td>...</td>
      <td> 17</td>
      <td> 25</td>
      <td> 39</td>
      <td> 121</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 0</td>
      <td> 0</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>  43</td>
      <td> 562</td>
      <td> 1</td>
      <td> 1</td>
      <td> 3</td>
      <td> 1</td>
      <td> 55</td>
      <td> 68</td>
      <td> 45</td>
      <td> 41</td>
      <td>...</td>
      <td>  8</td>
      <td> 15</td>
      <td> 24</td>
      <td> 150</td>
      <td> 6</td>
      <td> 6</td>
      <td> 7</td>
      <td> 4</td>
      <td> 0</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>80</th>
      <td> 100</td>
      <td> 525</td>
      <td> 2</td>
      <td> 0</td>
      <td> 0</td>
      <td> 3</td>
      <td> 62</td>
      <td> 32</td>
      <td> 38</td>
      <td> 18</td>
      <td>...</td>
      <td> 11</td>
      <td> 23</td>
      <td> 29</td>
      <td> 103</td>
      <td> 6</td>
      <td> 6</td>
      <td> 7</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>86</th>
      <td> 439</td>
      <td> 198</td>
      <td> 2</td>
      <td> 0</td>
      <td> 0</td>
      <td> 3</td>
      <td> 68</td>
      <td> 47</td>
      <td> 32</td>
      <td> 17</td>
      <td>...</td>
      <td> 11</td>
      <td> 17</td>
      <td> 21</td>
      <td> 108</td>
      <td> 6</td>
      <td> 7</td>
      <td> 6</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>7 </th>
      <td>   8</td>
      <td> 134</td>
      <td> 4</td>
      <td> 0</td>
      <td> 1</td>
      <td> 2</td>
      <td> 69</td>
      <td> 37</td>
      <td> 31</td>
      <td> 14</td>
      <td>...</td>
      <td>  4</td>
      <td> 16</td>
      <td> 12</td>
      <td>   0</td>
      <td> 6</td>
      <td> 5</td>
      <td> 6</td>
      <td> 0</td>
      <td> 0</td>
      <td> 5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>



# Now lets make our first prediction function: Say we want to predict the result of the matches. 
The match result is stored in 'Result' column of the input_data. The result is a discrete variable, 
i.e. it is either 1 or 0. So we will use Logistci regression. The sklearn library have linear_model module which proides
functions for Linear/Logistic regressions.


```python
from sklearn import linear_model
logistic_regr = linear_model.LogisticRegression()
```

We need to define input data and target data for the model. Out target data in this case will be data of the 'Result' column. And rest of the data will be input data. So lets seperate the input and target data.


```python
target_column = 'Result';
input_data_columns = [];
for column in input_data.keys():
    if column != target_column:
        input_data_columns.append(column);
```


```python
input_data_columns
```




    ['Player1',
     'Player2',
     'Round',
     'FNL.1',
     'FNL.2',
     'FSP.1',
     'FSW.1',
     'SSP.1',
     'SSW.1',
     'ACE.1',
     'DBF.1',
     'WNR.1',
     'UFE.1',
     'BPC.1',
     'BPW.1',
     'NPA.1',
     'NPW.1',
     'TPW.1',
     'ST1.1',
     'ST2.1',
     'ST3.1',
     'ST4.1',
     'ST5.1',
     'FSP.2',
     'FSW.2',
     'SSP.2',
     'SSW.2',
     'ACE.2',
     'DBF.2',
     'WNR.2',
     'UFE.2',
     'BPC.2',
     'BPW.2',
     'NPA.2',
     'NPW.2',
     'TPW.2',
     'ST1.2',
     'ST2.2',
     'ST3.2',
     'ST4.2',
     'ST5.2',
     'Type']




```python
data_x = input_data[input_data_columns];
data_y = input_data[target_column];
```

Lets train our classifier


```python
clf = logistic_regr.fit(data_x, data_y);
```

Now lets test this classifier on our test data


```python
test_data_x = test_data[input_data_columns];
predicted_result = clf.predict(test_data_x)
```


```python
predicted_result
```




    array([1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1,
           0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
           1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
           0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1,
           0, 0, 1, 0, 0, 1, 0, 1])



Now lets compare it with actual result of the matches.


```python
actual_result = test_data[target_column].values;
```

Lets make a function to compare the results


```python
def compare_data(predicted, actual):
    total_match = 0;
    total_mismatch = 0;
    for i in range(0, len(predicted)):
        if predicted[i] == actual[i]:
            total_match += 1;
        else:
            total_mismatch += 1;
    print 'Total tested: ' + str(total_match + total_mismatch);
    print 'Correct results: ' + str(total_match);
    print 'Incorrect results: ' + str(total_mismatch);
```


```python
compare_data(predicted_result, actual_result)
```

    Total tested: 100
    Correct results: 100
    Incorrect results: 0


Lets plot the data. We can use matplotlib module to plot the data


```python
%matplotlib inline
import matplotlib as plt
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
```


```python
test_data['Result'][:20].plot(kind='bar', subplots=True, figsize=(15, 5))
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x7fdbb2467ed0>], dtype=object)




![png](output_59_1.png)


To plot the predicted data we need to convert it into pandas data frame first.


```python
predicted_result = pd.DataFrame(predicted_result, columns=['Predicted Result']);
```


```python
predicted_result[:20].plot(kind='bar', subplots=True, figsize=(15, 5))
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x7fdbb23f3a10>], dtype=object)




![png](output_62_1.png)


We can see that both the graphs are exactly same.

# Now lets make another classifier which predicts how many Double Faults player one commits. 
The column which stores this data in input_data is 'DBF.1'. Since it is a continuous variable we will use linear 
regression this time. The target column will be 'DBF1' and rest of the columns will be input data for the classifier.


```python
target_column = 'DBF.1';
input_data_columns = [];
for column in input_data.keys():
    if column != target_column:
        input_data_columns.append(column);
```


```python
data_x = input_data[input_data_columns];
data_y = input_data[target_column];
```


```python
linear_regr = linear_model.LinearRegression()
```


```python
#Train the classifier
clf = linear_regr.fit(data_x, data_y);
```


```python
#Test data input
test_data_x = test_data[input_data_columns];

predicted_result = clf.predict(test_data_x)
```


```python
predicted_result
```




    array([ 3.96797089,  6.27319166,  2.05669611,  1.46838655,  2.24870816,
            3.58739409,  2.18509419,  3.82675637,  1.22865198,  3.78253087,
            5.04871872,  3.32239786,  2.43061685,  3.69821168,  4.48045371,
            4.37727147,  3.87743109,  2.11891937,  2.94163972,  5.1413804 ,
            1.77793897,  2.77361542,  3.08471639,  7.6246385 ,  3.31536512,
            7.98118011,  3.66015819,  2.64714928,  2.46350172,  4.64543521,
            3.59607087,  3.15755659,  1.81787362,  1.47703804,  1.68465481,
            1.88491107,  2.29791313,  5.26641278,  3.66195496,  5.54356756,
            5.63156975,  2.29442705,  4.41603268,  0.81141359,  2.62938263,
            4.02310272,  5.56898454,  1.80847002,  0.97380756,  5.96096087,
            2.04550562,  3.64602079,  4.83607019,  2.04871821,  3.79377401,
            1.28143892,  4.91983509,  1.40209716,  2.40962959,  6.26242749,
            3.83298518,  2.4323595 ,  4.99692163,  5.85918788,  3.41013109,
            4.66661767,  0.60592309,  3.97032148,  4.71272211,  0.9561459 ,
            1.98838691,  1.31824211,  3.49692729,  8.84060674,  4.43302787,
            1.34775195,  3.67089768,  2.21440687,  4.78220931,  4.30959574,
            0.55012598,  2.87770129,  5.4306943 ,  3.64387999,  3.27151726,
            3.37660302,  6.48117269,  2.80534254,  4.72867864,  3.71081455,
            4.65260386,  2.05506307,  3.51413563,  3.28651182,  6.02601975,
            4.00181255,  5.76838822,  3.67706674,  4.13482238,  5.51039093])




```python
test_data['Predicted_DBF.1'] = predicted_result
```


```python
#Lets check the actual result versus predicted result
test_data[['DBF.1', 'Predicted_DBF.1']]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DBF.1</th>
      <th>Predicted_DBF.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84 </th>
      <td> 5</td>
      <td> 3.967971</td>
    </tr>
    <tr>
      <th>73 </th>
      <td> 3</td>
      <td> 6.273192</td>
    </tr>
    <tr>
      <th>80 </th>
      <td> 3</td>
      <td> 2.056696</td>
    </tr>
    <tr>
      <th>86 </th>
      <td> 3</td>
      <td> 1.468387</td>
    </tr>
    <tr>
      <th>7  </th>
      <td> 1</td>
      <td> 2.248708</td>
    </tr>
    <tr>
      <th>79 </th>
      <td> 5</td>
      <td> 3.587394</td>
    </tr>
    <tr>
      <th>99 </th>
      <td> 2</td>
      <td> 2.185094</td>
    </tr>
    <tr>
      <th>121</th>
      <td> 1</td>
      <td> 3.826756</td>
    </tr>
    <tr>
      <th>68 </th>
      <td> 0</td>
      <td> 1.228652</td>
    </tr>
    <tr>
      <th>58 </th>
      <td> 6</td>
      <td> 3.782531</td>
    </tr>
    <tr>
      <th>17 </th>
      <td> 5</td>
      <td> 5.048719</td>
    </tr>
    <tr>
      <th>108</th>
      <td> 3</td>
      <td> 3.322398</td>
    </tr>
    <tr>
      <th>28 </th>
      <td> 1</td>
      <td> 2.430617</td>
    </tr>
    <tr>
      <th>78 </th>
      <td> 6</td>
      <td> 3.698212</td>
    </tr>
    <tr>
      <th>40 </th>
      <td> 8</td>
      <td> 4.480454</td>
    </tr>
    <tr>
      <th>32 </th>
      <td> 7</td>
      <td> 4.377271</td>
    </tr>
    <tr>
      <th>121</th>
      <td> 5</td>
      <td> 3.877431</td>
    </tr>
    <tr>
      <th>40 </th>
      <td> 1</td>
      <td> 2.118919</td>
    </tr>
    <tr>
      <th>67 </th>
      <td> 4</td>
      <td> 2.941640</td>
    </tr>
    <tr>
      <th>68 </th>
      <td> 8</td>
      <td> 5.141380</td>
    </tr>
    <tr>
      <th>122</th>
      <td> 1</td>
      <td> 1.777939</td>
    </tr>
    <tr>
      <th>38 </th>
      <td> 1</td>
      <td> 2.773615</td>
    </tr>
    <tr>
      <th>82 </th>
      <td> 6</td>
      <td> 3.084716</td>
    </tr>
    <tr>
      <th>41 </th>
      <td> 5</td>
      <td> 7.624638</td>
    </tr>
    <tr>
      <th>15 </th>
      <td> 5</td>
      <td> 3.315365</td>
    </tr>
    <tr>
      <th>45 </th>
      <td> 3</td>
      <td> 7.981180</td>
    </tr>
    <tr>
      <th>35 </th>
      <td> 3</td>
      <td> 3.660158</td>
    </tr>
    <tr>
      <th>24 </th>
      <td> 1</td>
      <td> 2.647149</td>
    </tr>
    <tr>
      <th>115</th>
      <td> 1</td>
      <td> 2.463502</td>
    </tr>
    <tr>
      <th>50 </th>
      <td> 4</td>
      <td> 4.645435</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41 </th>
      <td> 3</td>
      <td> 1.988387</td>
    </tr>
    <tr>
      <th>122</th>
      <td> 2</td>
      <td> 1.318242</td>
    </tr>
    <tr>
      <th>113</th>
      <td> 3</td>
      <td> 3.496927</td>
    </tr>
    <tr>
      <th>6  </th>
      <td> 8</td>
      <td> 8.840607</td>
    </tr>
    <tr>
      <th>35 </th>
      <td> 4</td>
      <td> 4.433028</td>
    </tr>
    <tr>
      <th>108</th>
      <td> 5</td>
      <td> 1.347752</td>
    </tr>
    <tr>
      <th>73 </th>
      <td> 2</td>
      <td> 3.670898</td>
    </tr>
    <tr>
      <th>93 </th>
      <td> 0</td>
      <td> 2.214407</td>
    </tr>
    <tr>
      <th>28 </th>
      <td> 6</td>
      <td> 4.782209</td>
    </tr>
    <tr>
      <th>85 </th>
      <td> 3</td>
      <td> 4.309596</td>
    </tr>
    <tr>
      <th>120</th>
      <td> 0</td>
      <td> 0.550126</td>
    </tr>
    <tr>
      <th>100</th>
      <td> 4</td>
      <td> 2.877701</td>
    </tr>
    <tr>
      <th>99 </th>
      <td> 8</td>
      <td> 5.430694</td>
    </tr>
    <tr>
      <th>44 </th>
      <td> 6</td>
      <td> 3.643880</td>
    </tr>
    <tr>
      <th>43 </th>
      <td> 2</td>
      <td> 3.271517</td>
    </tr>
    <tr>
      <th>31 </th>
      <td> 3</td>
      <td> 3.376603</td>
    </tr>
    <tr>
      <th>57 </th>
      <td> 1</td>
      <td> 6.481173</td>
    </tr>
    <tr>
      <th>1  </th>
      <td> 3</td>
      <td> 2.805343</td>
    </tr>
    <tr>
      <th>1  </th>
      <td> 2</td>
      <td> 4.728679</td>
    </tr>
    <tr>
      <th>96 </th>
      <td> 2</td>
      <td> 3.710815</td>
    </tr>
    <tr>
      <th>89 </th>
      <td> 8</td>
      <td> 4.652604</td>
    </tr>
    <tr>
      <th>27 </th>
      <td> 2</td>
      <td> 2.055063</td>
    </tr>
    <tr>
      <th>75 </th>
      <td> 1</td>
      <td> 3.514136</td>
    </tr>
    <tr>
      <th>77 </th>
      <td> 3</td>
      <td> 3.286512</td>
    </tr>
    <tr>
      <th>34 </th>
      <td> 5</td>
      <td> 6.026020</td>
    </tr>
    <tr>
      <th>36 </th>
      <td> 1</td>
      <td> 4.001813</td>
    </tr>
    <tr>
      <th>89 </th>
      <td> 9</td>
      <td> 5.768388</td>
    </tr>
    <tr>
      <th>70 </th>
      <td> 2</td>
      <td> 3.677067</td>
    </tr>
    <tr>
      <th>100</th>
      <td> 3</td>
      <td> 4.134822</td>
    </tr>
    <tr>
      <th>6  </th>
      <td> 3</td>
      <td> 5.510391</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>




```python
test_data[['DBF.1', 'Predicted_DBF.1']].plot(kind='bar', figsize=(15, 8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbaea856d0>




![png](output_73_1.png)



```python
test_data['Prediction_error'] = abs(test_data['DBF.1'] - test_data['Predicted_DBF.1'])*100/test_data['DBF.1']
```


```python
test_data['Prediction_error'].plot(kind='bar', figsize=(15, 8));
```


![png](output_75_0.png)


# Lets now predict the tournament name to which a game belongs to:

The data related to tournament name is in column 'Type'. We will predict it with Logistic regression, support vector machine and random forestal gorithms and will compare the results.


```python
#Our target column is 'Type'
target_column = 'Type';
input_data_columns = [];
for column in input_data.keys():
    if column != target_column:
        input_data_columns.append(column);
```

Based on the target column, lets get data_x, and data_y to train the classifier


```python
data_x = input_data[input_data_columns];
data_y = input_data[target_column];
```


```python
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#logistic regression classifier
logistic_clf = linear_model.LogisticRegression();
#Linear SVC classifier
svm_clf = svm.LinearSVC();
#Random forest classifier
forest_clf = RandomForestClassifier(n_estimators=160)
```

Lets train all the different classifiers now


```python
logistic_clf.fit(data_x, data_y);

svm_clf.fit(data_x, data_y);

forest_clf.fit(data_x, data_y);
```


```python
#Test data input
test_data_x = test_data[input_data_columns];
test_data_y = test_data[target_column];
```

Lets predict results for each classifier now


```python
logistic_predicted_results = logistic_clf.predict(test_data_x);
svm_predicted_results = svm_clf.predict(test_data_x);
forest_predicted_results = forest_clf.predict(test_data_x);
```

Lets put all the results in a single data frame for easy comparision


```python
results = pd.DataFrame({'Actual_Type': test_data_y.values,
                        'Logistic_Predicted_Type': logistic_predicted_results,
                        'Svm_Predicted_Type': svm_predicted_results,
                        'Forest_predicted_Type': forest_predicted_results
                       })
```


```python
#Lets see the predicted results versus actual result
results
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual_Type</th>
      <th>Forest_predicted_Type</th>
      <th>Logistic_Predicted_Type</th>
      <th>Svm_Predicted_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0 </th>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>1 </th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>2 </th>
      <td> 0</td>
      <td> 2</td>
      <td> 0</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>3 </th>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>4 </th>
      <td> 5</td>
      <td> 7</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>5 </th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>6 </th>
      <td> 1</td>
      <td> 3</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>7 </th>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>8 </th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>9 </th>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>10</th>
      <td> 7</td>
      <td> 7</td>
      <td> 5</td>
      <td> 5</td>
    </tr>
    <tr>
      <th>11</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>12</th>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>13</th>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>14</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>15</th>
      <td> 3</td>
      <td> 1</td>
      <td> 1</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>16</th>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>17</th>
      <td> 5</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>18</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>19</th>
      <td> 5</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>20</th>
      <td> 2</td>
      <td> 0</td>
      <td> 1</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>21</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>22</th>
      <td> 2</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>23</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>24</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>25</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>26</th>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>27</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>28</th>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>29</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>71</th>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>72</th>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>73</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>74</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>75</th>
      <td> 2</td>
      <td> 2</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>76</th>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>77</th>
      <td> 6</td>
      <td> 7</td>
      <td> 5</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>78</th>
      <td> 3</td>
      <td> 1</td>
      <td> 1</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>79</th>
      <td> 6</td>
      <td> 7</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>80</th>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>81</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>82</th>
      <td> 0</td>
      <td> 2</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>83</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>84</th>
      <td> 1</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>85</th>
      <td> 3</td>
      <td> 3</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>86</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>87</th>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>88</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>89</th>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>90</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>91</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>92</th>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>93</th>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>94</th>
      <td> 5</td>
      <td> 7</td>
      <td> 5</td>
      <td> 5</td>
    </tr>
    <tr>
      <th>95</th>
      <td> 1</td>
      <td> 1</td>
      <td> 3</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>96</th>
      <td> 2</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>97</th>
      <td> 1</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>98</th>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>99</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>



Number of correct predictions by Logistic regression classifier


```python
#Number of correct predictions
len(results[results['Actual_Type'] == results['Logistic_Predicted_Type']])
```




    67



Number of correct predictions by SVM classifier


```python
#Number of correct predictions
len(results[results['Actual_Type'] == results['Svm_Predicted_Type']])
```




    65



Number of correct predictions by Random forest classifier


```python
#Number of correct predictions
len(results[results['Actual_Type'] == results['Forest_predicted_Type']])
```




    70



Lets plot the first 20 results


```python
results[:20].plot(kind='bar', figsize=(15, 8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbae071750>




![png](output_96_1.png)


# Lets try Random forest algorithm on this data with different values of number of trees, and max_features

We will make three classifiers with therr different values of "max_features". Then we will plot the out of bag errors for each classifier for different number of trees. Based of the OOB errors we will select optimum pair of [max_features, trees].


```python
ensemble_clfs = [
    ("RF:max_features=sqrt",
        RandomForestClassifier(oob_score=True,
                               max_features="sqrt")),
    ("RF:max_features=log2",
        RandomForestClassifier(max_features='log2',
                               oob_score=True)),
    ("RF:max_features=None",
        RandomForestClassifier(max_features=None,
                               oob_score=True))
]
```


```python
from collections import OrderedDict
# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
error_rate['index'] = [];
```


```python
# Range of `n_estimators` values to explore.
min_estimators = 10
max_estimators = 200
```


```python
for i in range(min_estimators, max_estimators + 1):
    error_rate['index'].append(i)
    for label, clf in ensemble_clfs:
        clf.set_params(n_estimators=i)
        clf.fit(data_x, data_y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append(oob_error)
print len(error_rate['index'])
print len(error_rate['RF:max_features=sqrt'])
```

    191
    191



```python
errors = pd.DataFrame({'RF:max_features=sqrt': error_rate["RF:max_features=sqrt"],
                       'RF:max_features=log2': error_rate["RF:max_features=log2"],
                       'RF:max_features=None': error_rate["RF:max_features=None"],
                       'index': error_rate["index"],
                      })
errors = errors.set_index(['index'], drop=True, append=False, inplace=False, verify_integrity=False);
```


```python
errors.plot(figsize=(15,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbac0e8050>




![png](output_103_1.png)


Lets see the minimum error for each of the classifier, and trees


```python
#values of minimum errors
errors.min()
```




    RF:max_features=None    0.264846
    RF:max_features=log2    0.266033
    RF:max_features=sqrt    0.264846
    dtype: float64




```python
#Number of trees corresponding to the minimum errors
errors.idxmin()
```




    RF:max_features=None    136
    RF:max_features=log2    141
    RF:max_features=sqrt    196
    dtype: int64



So we see that minimum error for "max_features=None", and "max_features=sqrt" configuration is same and 
the corresponding number of trees are 136, and 196 respectvely.

So the optimum configuration for Random forest algorithm will be [max_features=sqrt, trees=196] 
or [max_features=None, trees=136].

Lets make a classifier with this configuration and predict the result on the data.


```python
forest_clf = RandomForestClassifier(max_features=None, n_estimators=136);
forest_clf.fit(data_x, data_y);
```


```python
predicted_results = forest_clf.predict(test_data_x);
```


```python
predicted_results
```




    array([3, 4, 0, 2, 7, 4, 3, 0, 6, 1, 5, 7, 2, 0, 6, 1, 3, 7, 6, 7, 2, 7, 0,
           6, 7, 4, 3, 6, 2, 5, 2, 4, 7, 2, 7, 1, 6, 2, 7, 5, 5, 0, 3, 3, 3, 1,
           4, 3, 3, 2, 6, 3, 6, 6, 6, 7, 1, 3, 3, 4, 7, 1, 4, 0, 2, 1, 7, 1, 6,
           7, 4, 3, 2, 4, 7, 0, 2, 6, 1, 7, 2, 4, 0, 6, 3, 3, 4, 2, 4, 1, 4, 7,
           2, 3, 7, 1, 0, 3, 1, 6])




```python
results['Forest_predicted_Type'] = predicted_results
```


```python
results
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual_Type</th>
      <th>Forest_predicted_Type</th>
      <th>Logistic_Predicted_Type</th>
      <th>Svm_Predicted_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0 </th>
      <td> 1</td>
      <td> 3</td>
      <td> 1</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>1 </th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>2 </th>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>3 </th>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>4 </th>
      <td> 5</td>
      <td> 7</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>5 </th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>6 </th>
      <td> 1</td>
      <td> 3</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>7 </th>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>8 </th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>9 </th>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>10</th>
      <td> 7</td>
      <td> 5</td>
      <td> 5</td>
      <td> 5</td>
    </tr>
    <tr>
      <th>11</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>12</th>
      <td> 0</td>
      <td> 2</td>
      <td> 0</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>13</th>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>14</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>15</th>
      <td> 3</td>
      <td> 1</td>
      <td> 1</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>16</th>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>17</th>
      <td> 5</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>18</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>19</th>
      <td> 5</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>20</th>
      <td> 2</td>
      <td> 2</td>
      <td> 1</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>21</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>22</th>
      <td> 2</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>23</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>24</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>25</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>26</th>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>27</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>28</th>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>29</th>
      <td> 7</td>
      <td> 5</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>71</th>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>72</th>
      <td> 0</td>
      <td> 2</td>
      <td> 2</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>73</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>74</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>75</th>
      <td> 2</td>
      <td> 0</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>76</th>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>77</th>
      <td> 6</td>
      <td> 6</td>
      <td> 5</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>78</th>
      <td> 3</td>
      <td> 1</td>
      <td> 1</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>79</th>
      <td> 6</td>
      <td> 7</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>80</th>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>81</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>82</th>
      <td> 0</td>
      <td> 0</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>83</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
    <tr>
      <th>84</th>
      <td> 1</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>85</th>
      <td> 3</td>
      <td> 3</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>86</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>87</th>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>88</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>89</th>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>90</th>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
      <td> 4</td>
    </tr>
    <tr>
      <th>91</th>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
      <td> 7</td>
    </tr>
    <tr>
      <th>92</th>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>93</th>
      <td> 1</td>
      <td> 3</td>
      <td> 1</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>94</th>
      <td> 5</td>
      <td> 7</td>
      <td> 5</td>
      <td> 5</td>
    </tr>
    <tr>
      <th>95</th>
      <td> 1</td>
      <td> 1</td>
      <td> 3</td>
      <td> 1</td>
    </tr>
    <tr>
      <th>96</th>
      <td> 2</td>
      <td> 0</td>
      <td> 0</td>
      <td> 2</td>
    </tr>
    <tr>
      <th>97</th>
      <td> 1</td>
      <td> 3</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>98</th>
      <td> 3</td>
      <td> 1</td>
      <td> 3</td>
      <td> 3</td>
    </tr>
    <tr>
      <th>99</th>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
      <td> 6</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>



Lets count the correct results now


```python
#Number of correct predictions
len(results[results['Actual_Type'] == results['Forest_predicted_Type']])
```




    69




```python

```
