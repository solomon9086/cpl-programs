import streamlit as st
st.title('Connecting to Python 3.11 Compiler')
st.subheader('making connection to server.......')
rad = st.selectbox('select',['','reverse number','Fabonacci','multiplication Table',
                         'remove space','Factorial','celcius to farenheit',
                         'distance','object oriented','traverse a tuple',
                         'graph plot','sine and cosine waves','odd index',
                         'square and cube','tkinter','dictionary','hypotenuse',
                         'common words','volume','2d list using pandas','sql connectivity',
                         'time series','df using list of tuples','pandas csv',
                         'jupyter','keyword and pos arg',
                         'pandas using np func'])

if rad == 'tkinter':
    code = """
import tkinter as tk
import ttkbootstrap as ttk

window = ttk.Window(themename="darkly")
window.title("Simple GUI")
window.geometry('400x200')  

def button_click():
    label.config(text="Button Clicked!")
button = ttk.Button(window, text="Click Me!", command=button_click)
label = ttk.Label(window, text="Hello!", font=("Calibri", 16))

button.pack(pady=20)  
label.pack()

window.mainloop()
    """

    st.code(code, language="python")
    
elif rad == 'reverse number':
    code = """
#reverse of a number
num = input('Enter number :')
print(num)
num2= ''
for i in range(1,len(num)+1):
    num2 = num2+num[-i] 

print(num2)
    """

    st.code(code, language="python")
    
elif rad == 'Fabonacci':
    code = """
#Fabonacci series
i = 0
j = 1
series = []
n = int(input('enter length of series : '))
for m in range(n):
    series.append(i)
    temp = i
    i = j
    j = j+temp
    
print(series)
    """

    st.code(code, language="python")
    
elif rad == 'multiplication Table':
    code = """
#multiplication Table
n = int(input('Enter n: '))

print('-----------')

for i in range(1,n+1):
    
    print(f'Multiplication Table of {i} is')
    
    for j in range(1,11):
        
        print(f'{i} * {j} = {i*j}')
        
    print('----------')
    """

    st.code(code, language="python")

elif rad == 'remove space':
    code = """
#remove space using continue
word = input('Enter : ')
word2 = ''

for i in range(len(word)):
    if word[i] == ' ':
        continue
    else:
        word2 = word2+word[i]

print(word2)

    """

    st.code(code, language="python")

elif rad == 'Factorial':
    code = """
#Factorial of a Number
number = int(input('Enter Number : '))
fact = 1
for i in range(1,number+1):
    fact = fact*i
    
print(f'Factorial of {number} is {fact}')
    """

    st.code(code, language="python")

elif rad == 'square and cube':
    code = """
# square and cube

num=int(input('Enter the number:'))
print('The square of number is:'+str(num**2))
print('The cube of number is:'+str(num**3))
    """

    st.code(code, language="python")
    
elif rad == 'distance':
    code = """
#Distance between two point
import math

p1 = (input('Enter (x1,y1) : '))
p2 = (input('Enter (x2,y2) : '))

p3 = p1.split(',')
p4 = p2.split(',')

x1 = int(p3[0])
y1 = int(p3[1])
x2 = int(p4[0])
y2 = int(p4[1])

dist = math.sqrt(((x2-x1)**2)+((y2-y1)**2))

print(f'Distance between ({x1},{y1}) & ({x2},{y2}) is {round(dist,4)}cm')
    """

    st.code(code, language="python")
    
elif rad == 'object oriented':
    code = """
#Program1 on oops
class student: 
    def __init__(self, name, age):
        self.name =name
        self.age =age
        
    def display_info(self):
        print(f"I am {self.name} and my age is {self.age}")
student1= student("abc",20)
student1.display_info()
    """

    st.code(code, language="python")
    
    code = """
#Program 2 \n Define a class called "Car"
class Car:

    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.speed = 0


    def accelerate(self):
        self.speed += 5

    def brake(self):
        if self.speed >= 5:
            self.speed -= 5
        else:
            self.speed = 0


    def display_info(self):
        return f"{self.year} {self.make} {self.model} traveling at {self.speed} mph"

car1 = Car("Toyota", "Camry", 2022)
car2 = Car("Honda", "Accord", 2021)

car1.accelerate()
car2.accelerate()
car1.accelerate()
car1.brake()

print("Car 1:", car1.display_info())
print("Car 2:", car2.display_info())

    """

    st.code(code, language="python")

elif rad == 'graph plot':
    code = """
#draw subplot
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-2,2,400)
y1=x
y2=x**2
y3=x**3
plt.figure(figsize=(10,6))

#y=x
plt.subplot(1,3,1)
plt.plot(x,y1,color='blue')
plt.title('y=x')

#y=x^2
plt.subplot(1,3,2)
plt.plot(x,y2,color='green')
plt.title('y=x^2')

#y=x^3
plt.subplot(1,3,3)
plt.plot(x,y3,color='red')
plt.title('y=x^3')

plt.tight_layout()
plt.show()
    """

    st.code(code, language="python")
    
elif rad == 'sine and cosine waves':
    code = """
#sine and cosine waves
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,2*np.pi,100)

y_sin=np.sin(x)
y_cos=np.cos(x)

plt.plot(x, y_sin, label='Sine Wave')
plt.plot(x, y_cos, label='Cosine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sine and Cosine Waves')
plt.legend()
plt.show()
    """
    st.code(code, language="python")
    
elif rad == 'celcius to farenheit':
    code = """
#convert celcius to farenheit
temp=[23,45,67,43,79]
for i in temp:
    f=(i*9/5)+32
    print(f)
    """
    st.code(code, language="python")
    
elif rad == 'traverse a tuple':
    code = """
#traverse a tuple
mark=[(75,67,98,84),(89,69,88),(78,98,67,69)]
for i in mark:
    for j in i:
        print(j)
    """
    st.code(code, language="python")
    
elif rad == 'odd index':
    code = """
#odd index into new tuple
T=(1,3,2,4,6,5)
list=[]
for i in T:
    while i%2==1:
        list.append(T[i])
        i=i+1
newtup= tuple(list)
print(newtup)
    """
    st.code(code, language="python")    

elif rad == 'printing lines':
    code = """
st.text('You done it bae')
    """
    st.code(code, language="python")  

elif rad == 'dictionary':
    code = """
def Histogram(word):
    empty_dict = {}
    for i in word:
        empty_dict[i] = empty_dict.get(i, 0) + 1
    return empty_dict

v = Histogram("AAPPLE")
print(v)
    """
    st.code(code, language="python")      

elif rad == 'hypotenuse':
    code = """
import math

print('consider a Right angle triangle ABC at B')

a = int(input('Enter length of AB :'))
b = int(input('Enter length of BC :'))

hyp = math.sqrt(a*2 + b*2)
print(f'hypotenuse : {round(hyp,2)}cm')
    """
    st.code(code, language="python")    

elif rad == 'common words':
    code = """
#common words
words1 = input('Enter 1st word : ')
words1 = words1.lower()
words2 = input('Enter 2nd word : ')
words2 = words2.lower()

words1 = words1.split()
words2 = words2.split()

common_words = set(words1) & set(words2)

print("Common words between word1 and word2:")
for word in common_words:
    print(word)
    """
    st.code(code, language="python")

elif rad == 'volume':
    code = """
#volume    
class Box:
    def _init_(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height
    def calculate_volume(self):
        volume = self.length * self.width * self.height
        return volume
length = float(input("Enter the length of the box: "))
width = float(input("Enter the width of the box: "))
height = float(input("Enter the height of the box: "))

my_box = Box(length, width, height)

volume = my_box.calculate_volume()

print(f"The volume of the box is: {volume} cubic unit")
    """
    st.code(code, language="python")  

elif rad == '2d list using pandas':
    code = """
# import pandas as pd 
import pandas as pd 
list = [['Alice', 25], ['Charlie', 30], 
	['John', 26], ['Bob', 22]] 
 
df = pd.DataFrame(list, columns =['Name', 'number']) 
print(df) 
    """
    st.code(code, language="python")      

elif rad == 'sql connectivity':
    code = """
#connectivity sql
import sqlite3

conn = sqlite3.connect('mydatabase.db')

cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY,
        name TEXT
    )
''')

cursor.execute("INSERT INTO students (name) VALUES ('Raj')")
cursor.execute("INSERT INTO students (name) VALUES ('Veer')")

cursor.execute("DELETE FROM students WHERE name = 'Veer'")

conn.commit()

cursor.execute('SELECT * FROM students')
data = cursor.fetchall()
for row in data:
    print(f"Student ID: {row[0]}, Name: {row[1]}")

conn.close()
    """
    st.code(code, language="python") 

elif rad == 'time series':
    code = """
#time series manipulation

import pandas as pd
from datetime import datetime
import numpy as np

range_date = pd.date_range(start ='1/1/2019', end ='1/08/2019',freq ='Min')

df = pd.DataFrame(range_date, columns =['date'])
df['data'] = np.random.randint(0, 100, size =(len(range_date)))

print(df.head(10))
    """
    st.code(code, language="python")      

elif rad == 'df using list of tuples':
    code = """
#pandas df using list of tuples    
import pandas as pd

data = [
    ('Alice', 25, 'Engineer'),
    ('Bob', 30, 'Data Scientist'),
    ('Charlie', 35, 'Designer'),
    ('Diana', 27, 'Developer'),
]
df = pd.DataFrame(data, columns=['Name', 'Age', 'Occupation'])
print(df)

    """
    st.code(code, language="python") 

elif rad == 'pandas csv':
    code = """
#pandas csv and missing data    
import pandas as pd

df = pd.read_csv('csv_location')  

missing_data = df.isnull()

df_cleaned = df.dropna()

df_filled = df.fillna('N/A')

df_interpolated = df.interpolate()

print("Original DataFrame:")
print(df)

print("\nDataFrame with missing data dropped:")
print(df_cleaned)

print("\nDataFrame with missing data filled:")
print(df_filled)

print("\nDataFrame with missing data interpolated:")
print(df_interpolated)
    """
    st.code(code, language="python")    

elif rad == 'pandas using np func':
    code = """
#pandas series using numpy functions
import pandas as pd 
import numpy as np 
    
ser1 = pd.Series(np.linspace(3, 33, 3)) 
print(ser1) 
  
ser2 = pd.Series(np.linspace(1, 100, 10)) 
print("\n", ser2)
    """
    st.code(code, language="python")



elif rad == 'jupyter':
    code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 20) 
y1 = x**2 
y2 = np.sin(x)  
data = {'x': x, 'y1': y1, 'y2': y2}


df = pd.DataFrame(data)

# Line plot
plt.figure(figsize=(8, 4))
plt.plot(df['x'], df['y1'], label='x^2', color='blue')
plt.plot(df['x'], df['y2'], label='sin(x)', color='red')
plt.title('Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

# Bar plot
plt.figure(figsize=(8, 4))
plt.bar(df['x'], df['y1'], label='x^2', color='green', alpha=0.5)
plt.title('Bar Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(df['x'], df['y2'], label='sin(x)', color='orange')
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
    """
    st.code(code, language="python")
    
elif rad == 'keyword and pos arg':
    code = """
def greet_message(name, message="Hello"):
    print(f"{message}, {name}!")

# Using positional arguments
greet_message("Ajay")  
greet_message("David", "Hi")  

# Using keyword arguments
greet_message(message="Hey", name="Raja")  
greet_message(name="pooja") 
    """
    st.code(code, language="python")