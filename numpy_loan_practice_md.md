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


```python
np.isnan(raw_data).sum()
```




    88005




```python
temp_fill = np.nanmax(raw_data)+1
temp_mean = np.nanmean(raw_data, axis = 0)
temp_mean
```

    /var/folders/vt/jvjts7bd7k7212z1cm0gmgzc0000gn/T/ipykernel_4710/1379539862.py:2: RuntimeWarning: Mean of empty slice
      temp_mean = np.nanmean(raw_data, axis = 0)





    array([54015809.19,         nan,    15273.46,         nan,    15311.04,         nan,       16.62,
                440.92,         nan,         nan,         nan,         nan,         nan,     3143.85])




```python
temp_stats = np.array([np.nanmin(raw_data, axis  = 0), temp_mean, np.nanmax(raw_data, axis = 0)])
```

    /var/folders/vt/jvjts7bd7k7212z1cm0gmgzc0000gn/T/ipykernel_4710/3419705878.py:1: RuntimeWarning: All-NaN slice encountered
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


```python
for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == ''), & (loan_data_strings[:,3] == i), i + '5', loan_data_strings[:,4])  
    
    
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




```python
Convert SubGrade to Numbers.
```


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


```python
State_Address
```


```python
np.unique(loan_data_strings[:,5]).size
```




    (50,)




```python

```
