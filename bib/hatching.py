import numpy as np
import numpy.ma as npma
import datetime
import csv

def get_mean_age(ages):
    #  Input: object array (ages of bee group)
    # Output: float (mean)
    if any(age is not None for age in ages):
        return np.mean(ages)
    else:
        return npma.masked_equal(ages, None).mean()
    
def get_std_age(ages):
    #  Input: object array (ages of bee group)
    # Output: float (standard deviation)
    if any(age is not None for age in ages):
        return np.std(ages)
    else:
        ages_edit = []
        for age in ages:
            if age is not None: 
                ages_edit.append(age)
        return np.std(ages_edit)

def get_bees_of_age(ages, age):
    #  Input: numpy object ndarray (age of all bees)
    # Output: numpy object ndarray (bees of specified age)    
    return np.where(ages == age)[0]

def get_all_bees_age(date):
    #  Input: date object
    # Output: numpy object ndarray (age of all bees)
    if isinstance(date, datetime.datetime):
        date = date.date()
    
    ages = []
    with open('hatchdates2016.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')	
        for row in reader:
            if reader.line_num == 1:
                continue
            if row[2] == '':
                age = None
            else:
                age = (date - str_to_datetime(row[2]).date()).days                
            ages.append(age)
    return np.array(ages)

def get_age(dec12, date):	
    #  Input: integer (12 o'clock, clockwise),
    #		  date object
    # Output: integer (age in days)
    if isinstance(date, datetime.datetime):
        date = date.date()
        
    if get_hatchdate(dec12) is None:
        return None
    return (date - get_hatchdate(dec12)).days

def get_hatchdate(dec12):
    #  Input: integer (12 o'clock, clockwise)
    # Output: date object
    f = pd.read_csv('hatchdates2016.csv')
    bday = f.ix[dec12].birthdate
    if (type(bday) != float):
        return str_to_datetime(bday).date()
    else:
        return None
    
def str_to_datetime(string):
    #  Input: string in format '%d.%m.%Y'
    # Output: datetime object
    if string == '':
        return None
    return datetime.datetime.strptime(string, '%d.%m.%Y')
    
def date_to_str(date):
    # Input: date(time) object
    # Output: string
    if date == None:
        return ''
    return date.strftime('%d.%m.%Y')


#if __name__ == '__main__':