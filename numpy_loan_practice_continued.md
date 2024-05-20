# Importing the Packages


```python
import numpy as np
```


```python
np.set_printoptions(suppress = True, linewidth = 100, precision = 2)

```

# Importing the data


```python
raw_data = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, autostrip = True, encoding = "unicode_escape")
raw_data
```




    array([[48010226.  ,         nan,    35000.  , ...,         nan,         nan,     9452.96],
           [57693261.  ,         nan,    30000.  , ...,         nan,         nan,     4679.7 ],
           [59432726.  ,         nan,    15000.  , ...,         nan,         nan,     1969.83],
           ...,
           [50415990.  ,         nan,    10000.  , ...,         nan,         nan,     2185.64],
           [46154151.  ,         nan,         nan, ...,         nan,         nan,     3199.4 ],
           [66055249.  ,         nan,    10000.  , ...,         nan,         nan,      301.9 ]])



# Checking for Incomplete Data
- We check for nan values in the dataset. Note that nan means - not a number, therefore may contain string data values.
- Seeing that there are 88005 nan values, we substitute them with the max+1 value (temp_fill).


```python
np.isnan(raw_data).sum()
```




    88005




```python
temp_fill = np.nanmax(raw_data)+1
temp_mean = np.nanmean(raw_data, axis = 0)
temp_mean
```

    /var/folders/vt/jvjts7bd7k7212z1cm0gmgzc0000gn/T/ipykernel_22607/1379539862.py:2: RuntimeWarning: Mean of empty slice
      temp_mean = np.nanmean(raw_data, axis = 0)





    array([54015809.19,         nan,    15273.46,         nan,    15311.04,         nan,       16.62,
                440.92,         nan,         nan,         nan,         nan,         nan,     3143.85])




```python
temp_stats = np.array([np.nanmin(raw_data, axis  = 0), temp_mean, np.nanmax(raw_data, axis = 0)])
```

    /var/folders/vt/jvjts7bd7k7212z1cm0gmgzc0000gn/T/ipykernel_22607/3419705878.py:1: RuntimeWarning: All-NaN slice encountered
      temp_stats = np.array([np.nanmin(raw_data, axis  = 0), temp_mean, np.nanmax(raw_data, axis = 0)])



```python
temp_stats
```




    array([[  373332.  ,         nan,     1000.  ,         nan,     1000.  ,         nan,        6.  ,
                  31.42,         nan,         nan,         nan,         nan,         nan,        0.  ],
           [54015809.19,         nan,    15273.46,         nan,    15311.04,         nan,       16.62,
                 440.92,         nan,         nan,         nan,         nan,         nan,     3143.85],
           [68616519.  ,         nan,    35000.  ,         nan,    35000.  ,         nan,       28.99,
                1372.97,         nan,         nan,         nan,         nan,         nan,    41913.62]])



# Splitting the dataset
We Split the dataset, separating string data from numeric data.
The argwhere fxn is used in combination with the isnan fxn to locate the indices of the columns with nan values (string) or num values.


```python
col_strings = np.argwhere(np.isnan(temp_mean)).squeeze()
```


```python
col_strings
```




    array([ 1,  3,  5,  8,  9, 10, 11, 12])




```python
col_num = np.argwhere(np.isnan(temp_mean) == False).squeeze()
```


```python
col_num
```




    array([ 0,  2,  4,  6,  7, 13])



# Re Importing the DataSet
Here we reimport the splitted dataset (numeric and string) as separate files.


```python
loan_data_strings = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, autostrip = True,
                                  usecols = col_strings, encoding = "unicode_escape", dtype = np.str_)
```


```python
loan_data_strings
```




    array([['May-15', 'Current', '36 months', ..., 'Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=48010226', 'CA'],
           ['', 'Current', '36 months', ..., 'Source Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=57693261', 'NY'],
           ['Sep-15', 'Current', '36 months', ..., 'Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=59432726', 'PA'],
           ...,
           ['Jun-15', 'Current', '36 months', ..., 'Source Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=50415990', 'CA'],
           ['Apr-15', 'Current', '36 months', ..., 'Source Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=46154151', 'OH'],
           ['Dec-15', 'Current', '36 months', ..., '',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=66055249', 'IL']],
          dtype='<U69')




```python
loan_data_num = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, autostrip = True,
                                  usecols = col_num, encoding = "unicode_escape", filling_values = temp_fill)
```


```python
loan_data_num
```




    array([[48010226.  ,    35000.  ,    35000.  ,       13.33,     1184.86,     9452.96],
           [57693261.  ,    30000.  ,    30000.  , 68616520.  ,      938.57,     4679.7 ],
           [59432726.  ,    15000.  ,    15000.  , 68616520.  ,      494.86,     1969.83],
           ...,
           [50415990.  ,    10000.  ,    10000.  , 68616520.  , 68616520.  ,     2185.64],
           [46154151.  , 68616520.  ,    10000.  ,       16.55,      354.3 ,     3199.4 ],
           [66055249.  ,    10000.  ,    10000.  , 68616520.  ,      309.97,      301.9 ]])



# Storing the names of the columns (Headers)
Here we store the full header list, string header and numeric header in different variable names.


```python
header_full = np.genfromtxt("loan-data.csv", delimiter = ';', skip_footer = raw_data.shape[0], autostrip = True,
                                   encoding = "unicode_escape", dtype = np.str_)
```


```python
header_full 
```




    array(['id', 'issue_d', 'loan_amnt', 'loan_status', 'funded_amnt', 'term', 'int_rate',
           'installment', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state',
           'total_pymnt'], dtype='<U19')




```python
string_headers, num_headers = header_full[col_strings], header_full[col_num]
```


```python
string_headers
```




    array(['issue_d', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url',
           'addr_state'], dtype='<U19')




```python
num_headers
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')



# Creating Checkpoints
Here, checkpoints are used to store a copy of our dataset to avoild loosing all the progress made.


```python
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)
```


```python
checkpoint_test = checkpoint("checkpoint_test", string_headers, loan_data_strings)
```


```python
checkpoint_test['data']
```




    array([['May-15', 'Current', '36 months', ..., 'Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=48010226', 'CA'],
           ['', 'Current', '36 months', ..., 'Source Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=57693261', 'NY'],
           ['Sep-15', 'Current', '36 months', ..., 'Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=59432726', 'PA'],
           ...,
           ['Jun-15', 'Current', '36 months', ..., 'Source Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=50415990', 'CA'],
           ['Apr-15', 'Current', '36 months', ..., 'Source Verified',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=46154151', 'OH'],
           ['Dec-15', 'Current', '36 months', ..., '',
            'https://www.lendingclub.com/browse/loanDetail.action?loan_id=66055249', 'IL']],
          dtype='<U69')



# Manipulating String Columns

## Issue Date
Here we strip the common parts of  the string "-15" and also store the swap the months Jan - Dec with numbers 1-12


```python
string_headers[0] = "issue_date"
```


```python
np.unique(loan_data_strings[:,0])
```




    array(['', 'Apr-15', 'Aug-15', 'Dec-15', 'Feb-15', 'Jan-15', 'Jul-15', 'Jun-15', 'Mar-15',
           'May-15', 'Nov-15', 'Oct-15', 'Sep-15'], dtype='<U69')




```python
loan_data_strings[:,0] = np.chararray.strip(loan_data_strings[:,0], "-15")
```


```python
np.unique(loan_data_strings[:,0])
```




    array(['', 'Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep'],
          dtype='<U69')




```python
months = np.array(['','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
```


```python
months
```




    array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
          dtype='<U3')




```python
for i in range(13):
    loan_data_strings[:,0] = np.where(loan_data_strings[:,0] == months[i], i,loan_data_strings[:,0])
```


```python
np.unique(loan_data_strings[:,0])
```




    array(['0', '1', '10', '11', '12', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U69')



## Loan-status
Here, the loan status values are categorized into two: status_bad (charged_off, default, empty, Late 31-120 days) and status_good.
status_bad values are changed to '0' and good to '1'


```python
np.unique(loan_data_strings[:,1])
```




    array(['', 'Charged Off', 'Current', 'Default', 'Fully Paid', 'In Grace Period', 'Issued',
           'Late (16-30 days)', 'Late (31-120 days)'], dtype='<U69')




```python
status_bad = np.array(['', 'Charged Off', 'Default', 'Late (31-120 days)'])
```


```python
loan_data_strings[:,1] = np.where(np.isin(loan_data_strings[:,1], status_bad),0,1)
```


```python
np.unique(loan_data_strings[:,1])
```




    array(['0', '1'], dtype='<U69')



## Term
Here, the common string 'months' is stripped off.
Also, we assign 60 to any empty cell (following normal convention to ascribe to worst possible outcome to empty cells).



```python
np.unique(loan_data_strings[:,2])
```




    array(['', '36 months', '60 months'], dtype='<U69')




```python
loan_data_strings[:,2] = np.chararray.strip(loan_data_strings[:,2], " months")
loan_data_strings[:,2]
```




    array(['36', '36', '36', ..., '36', '36', '36'], dtype='<U69')




```python
loan_data_strings[:,2] = np.where(loan_data_strings[:,2] == '', '60', loan_data_strings[:,2])

```

## Grades & Subgrades
- Here we substitue the empty spaces  in sub_grade with the worst possible outcome of its grade. i.e if grade == B, empty sub_grade == B5.
- Also, we notice that there are individuals with neither Grade nor sub_grade, hence we fill the sun_grades of these cells with a value of H1(worst possible outcome).
- Since we no longer need the Grades column, we delete it as well its header.


```python
np.unique(loan_data_strings[:,3])
np.unique(loan_data_strings[:,3])
```




    array(['', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='<U69')




```python
for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') & (loan_data_strings[:,3] == i), i + '5', loan_data_strings[:,4])  
    
    
```


```python
for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where(loan_data_strings[:,4] == '', 'H1', loan_data_strings[:,4])  
    
    
```


```python
np.unique(loan_data_strings[:,4])
```




    array(['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5',
           'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5',
           'G1', 'G2', 'G3', 'G4', 'G5', 'H1'], dtype='<U69')




```python
loan_data_strings = np.delete(loan_data_strings, 3, axis =1)
```


```python
string_headers = np.delete(string_headers, 3)
```


```python
string_headers[3]
```




    'sub_grade'



### CONVERTING SUB_GRADE
As before, we swap the unique subgrades with numeric values.


```python
sub_array = (['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5',
       'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5',
       'G1', 'G2', 'G3', 'G4', 'G5', 'H1'])
```


```python
sub_array[1]
```




    'A2'



## Convert SubGrade to Numbers.


```python
for i in range(1,37):
    loan_data_strings[:,3] = np.where(loan_data_strings[:,3] == sub_array[i-1], i,loan_data_strings[:,3])
```


```python
np.unique(loan_data_strings[:,3])
```




    array(['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22',
           '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36',
           '4', '5', '6', '7', '8', '9'], dtype='<U69')



## Verification 
As usual, we fill empty space with worse case(not verified).
Also, we categorize the data into good and bad and set the good to 1 and bad to 0


```python
loan_data_strings[:,4]
```




    array(['Verified', 'Source Verified', 'Verified', ..., 'Source Verified', 'Source Verified', ''],
          dtype='<U69')




```python
loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') | (loan_data_strings[:,4] == 'Not Verified'), 0,1)
```

## URL
-The repeating strings are stripped.
Eventually, we see that the URL column is redundant, so we delete it.


```python
loan_data_strings[:,5] = np.chararray.strip(loan_data_strings[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")
```


```python
loan_data_strings = np.delete(loan_data_strings, 5, axis =1)
string_headers = np.delete(string_headers, 5)
```

## State_Address
Here, we see that the dataset contains about 50 states in abbreviations. Also we see that the number of loan applicants per states is skewed such that some staes are lowly represented.
-Hence, to avoid outliers in number allocation, we group the states with a common characteristic- Region.
-Then we substitute the states in these categories with numbers ranging 1-4.
-Entries with missing states are assigned the number 0.


```python
np.unique(loan_data_strings[:,5])
```




    array(['', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IL', 'IN',
           'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH',
           'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA',
           'VT', 'WA', 'WI', 'WV', 'WY'], dtype='<U69')




```python
string_headers
```




    array(['issue_date', 'loan_status', 'term', 'sub_grade', 'verification_status', 'addr_state'],
          dtype='<U19')




```python
string_headers[5] = "state_address"
```


```python
states_names, states_count = np.unique(loan_data_strings[:,5], return_counts = True)
states_count_sorted = np.argsort(-states_count)
states_names[states_count_sorted], states_count[states_count_sorted]
```




    (array(['CA', 'NY', 'TX', 'FL', '', 'IL', 'NJ', 'GA', 'PA', 'OH', 'MI', 'NC', 'VA', 'MD', 'AZ',
            'WA', 'MA', 'CO', 'MO', 'MN', 'IN', 'WI', 'CT', 'TN', 'NV', 'AL', 'LA', 'OR', 'SC', 'KY',
            'KS', 'OK', 'UT', 'AR', 'MS', 'NH', 'NM', 'WV', 'HI', 'RI', 'MT', 'DE', 'DC', 'WY', 'AK',
            'NE', 'SD', 'VT', 'ND', 'ME'], dtype='<U69'),
     array([1336,  777,  758,  690,  500,  389,  341,  321,  320,  312,  267,  261,  242,  222,  220,
             216,  210,  201,  160,  156,  152,  148,  143,  143,  130,  119,  116,  108,  107,   84,
              84,   83,   74,   74,   61,   58,   57,   49,   44,   40,   28,   27,   27,   27,   26,
              25,   24,   17,   16,   10]))




```python
loan_data_strings[:,5] = np.where(loan_data_strings[:,5] == '',0, loan_data_strings[:,5])
```


```python
states_west = np.array(['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY'])
states_south = np.array(['AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV','DC'])
states_midwest = np.array(['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'])
states_east = np.array(['CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT'])
```


```python
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_west),1, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_south),2, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_midwest),3, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_east),4, loan_data_strings[:,5])
```


```python
np.unique(loan_data_strings[:,5])
```




    array(['0', '1', '2', '3', '4'], dtype='<U69')



## Converting to Numbers
Here we cast all the numbers we allocated to the string data int the integer data type.


```python
np.unique(loan_data_strings)
```




    array(['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22',
           '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36',
           '4', '5', '6', '60', '7', '8', '9'], dtype='<U69')




```python
loan_data_strings = loan_data_strings.astype(np.int16)
```


```python
loan_data_strings
```




    array([[ 5,  1, 36, 13,  1,  1],
           [ 0,  1, 36,  5,  1,  4],
           [ 9,  1, 36, 10,  1,  4],
           ...,
           [ 6,  1, 36,  5,  1,  1],
           [ 4,  1, 36, 17,  1,  3],
           [12,  1, 36,  4,  0,  3]], dtype=int16)



## Creating Checkpoints


```python
checkpoint_strings = checkpoint("Checkpoint-Strings", string_headers, loan_data_strings)
```


```python
checkpoint_strings['data']
```




    array([[ 5,  1, 36, 13,  1,  1],
           [ 0,  1, 36,  5,  1,  4],
           [ 9,  1, 36, 10,  1,  4],
           ...,
           [ 6,  1, 36,  5,  1,  1],
           [ 4,  1, 36, 17,  1,  3],
           [12,  1, 36,  4,  0,  3]], dtype=int16)



## Manipulating Numeric Columns
### Substitute 'Filler' Values
-Remember we filled the missing numeric values with the max value (temp_fill)?
-Now, we need to substitue those fillers with worst possible outcome for each column.
-We check that the id column (the primary column) does not contain filler values.
-Considering the diifernt columns, id ....total_pymnt, only the 'funded_amnt' column may need to be substituted to the minimum value as the worst possible outcome. This done by substituting any temp_fill value present in the 'funded_amnt' column with the nanmin which we derived from the temp_stats variable above.



```python
loan_data_num
```




    array([[48010226.  ,    35000.  ,    35000.  ,       13.33,     1184.86,     9452.96],
           [57693261.  ,    30000.  ,    30000.  , 68616520.  ,      938.57,     4679.7 ],
           [59432726.  ,    15000.  ,    15000.  , 68616520.  ,      494.86,     1969.83],
           ...,
           [50415990.  ,    10000.  ,    10000.  , 68616520.  , 68616520.  ,     2185.64],
           [46154151.  , 68616520.  ,    10000.  ,       16.55,      354.3 ,     3199.4 ],
           [66055249.  ,    10000.  ,    10000.  , 68616520.  ,      309.97,      301.9 ]])




```python
np.isnan(loan_data_num).sum()
```




    0




```python
num_headers
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')




```python
temp_fill
```




    68616520.0




```python
np.isin(loan_data_num[:,0], temp_fill).sum()
```




    0




```python
temp_stats[:, col_num]
```




    array([[  373332.  ,     1000.  ,     1000.  ,        6.  ,       31.42,        0.  ],
           [54015809.19,    15273.46,    15311.04,       16.62,      440.92,     3143.85],
           [68616519.  ,    35000.  ,    35000.  ,       28.99,     1372.97,    41913.62]])




```python
loan_data_num[:,2] = np.where(loan_data_num[:,2] == temp_fill, temp_stats[0, col_num[2]],loan_data_num[:,2])
```

### Loan_Amnt,Int_Rate, Total_pymnt, Installment
Here, we also substitute the temp_fill (which poses an outlier risk) to the max value in each of the respective columns.


```python
num_headers
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')




```python
for i in [1,3,4,5]:
    loan_data_num[:,i] = np.where(loan_data_num[:,i] == temp_fill, temp_stats[2, col_num[i]],loan_data_num[:,i])
```

## Currency Change
### The Exchange Rate
-We need to keep the exchange rates in euro as well as dollars.
We import the Eur-Usd file which contains the average monthly exchange rate for 2015.
We use only the closing exchange rate in the 4th column.
-We store the exchange rate for each individual in Loan-data according to the month of the transaction.
-For the accounts/individuals with no issue dates '0', we assign the annual exchange rate.


```python
EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter = ',', autostrip = True, dtype =np.str_)
EUR_USD
```




    array([['Open', 'High', 'Low', 'Close', 'Volume'],
           ['1.2098628282546997', '1.2098628282546997', '1.11055588722229', '1.1287955045700073', '0'],
           ['1.1287955045700073', '1.1484194993972778', '1.117680549621582', '1.1205360889434814',
            '0'],
           ['1.119795799255371', '1.1240400075912476', '1.0460032224655151', '1.0830246210098267',
            '0'],
           ['1.0741022825241089', '1.1247594356536865', '1.0521597862243652', '1.1114321947097778',
            '0'],
           ['1.1215037107467651', '1.145304799079895', '1.0821995735168457', '1.0960345268249512',
            '0'],
           ['1.095902442932129', '1.1428401470184326', '1.0888904333114624', '1.122296690940857', '0'],
           ['1.1134989261627197', '1.1219995021820068', '1.081270456314087', '1.0939244031906128',
            '0'],
           ['1.0969001054763794', '1.1705996990203857', '1.0850305557250977', '1.1340054273605347',
            '0'],
           ['1.1225990056991577', '1.1460003852844238', '1.1089695692062378', '1.1255937814712524',
            '0'],
           ['1.1171561479568481', '1.1494200229644775', '1.0910003185272217', '1.100897192955017',
            '0'],
           ['1.1024993658065796', '1.1060001850128174', '1.056400179862976', '1.0583018064498901',
            '0'],
           ['1.0572947263717651', '1.107000470161438', '1.0541995763778687', '1.093398094177246', '0']],
          dtype='<U18')




```python
EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter = ',', autostrip = True, skip_header = 1, usecols = 3)
```


```python
EUR_USD
```




    array([1.13, 1.12, 1.08, 1.11, 1.1 , 1.12, 1.09, 1.13, 1.13, 1.1 , 1.06, 1.09])




```python
np.unique(loan_data_strings[:,0])
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int16)




```python
exchange_rate = loan_data_strings[:,0]
for i in range(1,13):
    exchange_rate = np.where(exchange_rate == i, EUR_USD[i-1], exchange_rate)

exchange_rate = np.where(exchange_rate == 0, np.mean(EUR_USD), exchange_rate)
exchange_rate
```




    array([1.1 , 1.11, 1.13, ..., 1.12, 1.11, 1.09])




```python
exchange_rate.shape

```




    (10000,)




```python
loan_data_num.shape
```




    (10000, 6)



### We try to match the shapes of both arrays to enable us to add the exchange_rate array to the numeric column.
We also add the exchange rate to the list of headers.


```python
exchange_rate = np.reshape(exchange_rate, (10000,1))

```


```python
loan_data_num = np.hstack((loan_data_num, exchange_rate))
```


```python
exchange_rate
```




    array([[1.1 ],
           [1.11],
           [1.13],
           ...,
           [1.12],
           [1.11],
           [1.09]])




```python
num_headers = np.concatenate((num_headers,'exchange_rate'))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[86], line 1
    ----> 1 num_headers = np.concatenate((num_headers,'exchange_rate'))


    ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 0 dimension(s)


### Since the shapes do not match, we create a 1D array for exchange_rate


```python
num_headers = np.concatenate((num_headers, np.array(['exchange_rate'])))
```

### From USD to EUR
Here, we take the USD amounts in the necessary columns [1,2,4,5], divide them by the exchange_rate to get their EUR equivalent and stack/add the new EUR amounts  column to the loan_data_num dataset.


```python
num_headers
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt', 'exchange_rate',
           'exchange_rate', 'exchange_rate'], dtype='<U19')




```python
columns_dollar = np.array([1,2,4,5])
```


```python
loan_data_num[:,[columns_dollar]]
```




    array([[[35000.  , 35000.  ,  1184.86,  9452.96]],
    
           [[30000.  , 30000.  ,   938.57,  4679.7 ]],
    
           [[15000.  , 15000.  ,   494.86,  1969.83]],
    
           ...,
    
           [[10000.  , 10000.  ,  1372.97,  2185.64]],
    
           [[35000.  , 10000.  ,   354.3 ,  3199.4 ]],
    
           [[10000.  , 10000.  ,   309.97,   301.9 ]]])




```python
for i in columns_dollar:
    loan_data_num = np.hstack((loan_data_num, loan_data_num[:,i]/loan_data_num[:,6]))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[93], line 2
          1 for i in columns_dollar:
    ----> 2     loan_data_num = np.hstack((loan_data_num, loan_data_num[:,i]/loan_data_num[:,6]))


    File /opt/anaconda3/lib/python3.11/site-packages/numpy/core/shape_base.py:359, in hstack(tup, dtype, casting)
        357     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
        358 else:
    --> 359     return _nx.concatenate(arrs, 1, dtype=dtype, casting=casting)


    ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 1 dimension(s)



```python
for i in columns_dollar:
    loan_data_num = np.hstack((loan_data_num, np.reshape(loan_data_num[:,i]/loan_data_num[:,6],(10000,1))))
```

### Expanding the 
-We try to create headers for the new EUR values we have generated.
-We also try to add the 'USD' to the old amt headers to differentiate from the EUR.
_Lastly, we will rearrange the columns so that each EUR column follows its corresponding USD column.


```python
header_additional = np.array([column_name + '_EUR' for column_name in num_headers[columns_dollar]])
```


```python
header_additional
```




    array(['loan_amnt_EUR', 'funded_amnt_EUR', 'installment_EUR', 'total_pymnt_EUR'], dtype='<U15')




```python
num_headers = np.concatenate((num_headers, header_additional))
```


```python
num_headers
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt', 'exchange_rate',
           'exchange_rate', 'exchange_rate', 'loan_amnt_EUR', 'funded_amnt_EUR', 'installment_EUR',
           'total_pymnt_EUR'], dtype='<U19')




```python
num_headers[columns_dollar] = np.array([column_name + '_USD' for column_name in num_headers[columns_dollar]])
```


```python
num_headers
```




    array(['id', 'loan_amnt_USD', 'funded_amnt_USD', 'int_rate', 'installment_USD', 'total_pymnt_USD',
           'exchange_rate', 'exchange_rate', 'exchange_rate', 'loan_amnt_EUR', 'funded_amnt_EUR',
           'installment_EUR', 'total_pymnt_EUR'], dtype='<U19')




```python
columns_index_order = [0,1,7,2,8,3,4,9,5,10,6]
```


```python
num_headers = num_headers[columns_index_order]
```


```python
loan_data_num = loan_data_num[:,columns_index_order]
```

### Interest Rate
We divide all int.rates by 100 to make the values fall between 0 and 1 for easier analysis.


```python
num_headers
```




    array(['id', 'loan_amnt_USD', 'exchange_rate', 'funded_amnt_USD', 'exchange_rate', 'int_rate',
           'installment_USD', 'loan_amnt_EUR', 'total_pymnt_USD', 'funded_amnt_EUR', 'exchange_rate'],
          dtype='<U19')




```python
loan_data_num[:,5]
```




    array([13.33, 28.99, 28.99, ..., 28.99, 16.55, 28.99])




```python
loan_data_num[:,5] = loan_data_num[:,5]/100
```


```python
loan_data_num[:,5]
```




    array([0.13, 0.29, 0.29, ..., 0.29, 0.17, 0.29])



### Checkpoint 2: Numeric



```python
checkpoint_numeric = checkpoint("Checkpoint-Numeric", num_headers, loan_data_num)
```

## Creating the 'Complete" Dataset
Here we combine the split datasets together: strings + numeric.
But first, we have to ensure that the two datasets have same shape.


```python
checkpoint_strings['data'].shape
```




    (10000, 6)




```python
checkpoint_numeric['data'].shape
```




    (10000, 11)




```python
loan_data = np.hstack((checkpoint_numeric['data'],checkpoint_strings['data']))
```


```python
loan_data
```




    array([[48010226.  ,    35000.  ,    31933.3 , ...,       13.  ,        1.  ,        1.  ],
           [57693261.  ,    30000.  ,    27132.46, ...,        5.  ,        1.  ,        4.  ],
           [59432726.  ,    15000.  ,    13326.3 , ...,       10.  ,        1.  ,        4.  ],
           ...,
           [50415990.  ,    10000.  ,     8910.3 , ...,        5.  ,        1.  ,        1.  ],
           [46154151.  ,    35000.  ,    31490.9 , ...,       17.  ,        1.  ,        3.  ],
           [66055249.  ,    10000.  ,     9145.8 , ...,        4.  ,        0.  ,        3.  ]])




```python
np.isnan(loan_data).sum()
```




    0




```python
header_full = np.concatenate((checkpoint_numeric['header'],checkpoint_strings['header']))
```

### Sorting the New DataSet
We sort the 'Complete' Dataset according to the ID column.


```python
loan_data = loan_data[np.argsort(loan_data[:,0])]
```


```python
loan_data
```




    array([[  373332.  ,     9950.  ,     9038.08, ...,       21.  ,        0.  ,        1.  ],
           [  575239.  ,    12000.  ,    10900.2 , ...,       25.  ,        1.  ,        2.  ],
           [  707689.  ,    10000.  ,     8924.3 , ...,       13.  ,        1.  ,        0.  ],
           ...,
           [68614880.  ,     5600.  ,     5121.65, ...,        8.  ,        1.  ,        1.  ],
           [68615915.  ,     4000.  ,     3658.32, ...,       10.  ,        1.  ,        2.  ],
           [68616519.  ,    21600.  ,    19754.93, ...,        3.  ,        0.  ,        2.  ]])



### Storing the New DataSet
We stack the new header (header_full) to the Complete dataset (loan_data)


```python
loan_data = np.vstack((header_full, loan_data))
```


```python
loan_data
```




    array([['id', 'loan_amnt_USD', 'exchange_rate', ..., 'sub_grade', 'verification_status',
            'state_address'],
           ['373332.0', '9950.0', '9038.082814338286', ..., '21.0', '0.0', '1.0'],
           ['575239.0', '12000.0', '10900.20037910145', ..., '25.0', '1.0', '2.0'],
           ...,
           ['68614880.0', '5600.0', '5121.647851612413', ..., '8.0', '1.0', '1.0'],
           ['68615915.0', '4000.0', '3658.319894008867', ..., '10.0', '1.0', '2.0'],
           ['68616519.0', '21600.0', '19754.927427647883', ..., '3.0', '0.0', '2.0']], dtype='<U32')




```python
np.savetxt("loan-data-preprocessed.csv", loan_data, fmt = "%s", delimiter = ',')
```
