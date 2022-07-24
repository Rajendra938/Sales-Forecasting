#Project by Rajendra, Date: 20-06-2022


#importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import urllib
import pickle


def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Rajendra938/Sales-Forecasting/main/app/' + path
    
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

readme_text = st.markdown(get_file_content_as_string('instruction.md'))

#loading the files in cache to save time
@st.cache
def load_center():
    center_meta = pd.read_csv('fulfilment_center_info.csv')
    center_meta= center_meta.sort_values('center_id')
    center_meta = center_meta.reset_index()
    del center_meta['index']
    return center_meta
@st.cache
def load_database():
    data = pd.read_csv('database.csv')
    return data

@st.cache
def load_meal():
    meal_meta = pd.read_csv('meal_info.csv')
    meal_meta= meal_meta.sort_values('meal_id')
    meal_meta = meal_meta.reset_index()
    del meal_meta['index']
    return meal_meta


#Function for feature engineering 
def fea(df,test):
    #df = database
    #test = test query point
        
        temp =df['num_orders'].loc[(df['meal_id'] == int(test.meal_id)) & (df['center_id'] == int(test.center_id)) &  (df['week'] ==int((test['week']-1)))]
        #print (temp)
        temp = np.array(temp)
        if len(temp)!=0:
            test['Past1']= temp
        else:
            test['Past1']=136
            
        temp =df['num_orders'].loc[(df['meal_id'] == int(test.meal_id)) & (df['center_id'] == int(test.center_id)) &  (df['week'] ==int((test['week']-2)))]
        temp = np.array(temp)
        if len(temp)!=0:
            test['Past2']= temp
        else:
            test['Past2']=136
        
        temp =df['num_orders'].loc[(df['meal_id'] == int(test.meal_id)) & (df['center_id'] == int(test.center_id)) &  (df['week'] ==int((test['week']-3)))]
        #print (temp) 
        temp = np.array(temp)
        if len(temp)!=0:
            test['Past3']= temp
        else:
            test['Past3']=136
        
        temp =df['num_orders'].loc[(df['meal_id'] == int(test.meal_id)) & (df['center_id'] == int(test.center_id)) &  (df['week'] ==int((test['week']-4)))]
        #print (temp)
        temp = np.array(temp)
        if len(temp)!=0:
            test['Past4']= temp
        else:
            test['Past4']=136
            
        temp =df['num_orders'].loc[(df['meal_id'] == int(test.meal_id)) & (df['center_id'] == int(test.center_id)) &  (df['week'] ==int((test['week']-5)))]
        #print (temp)
        temp = np.array(temp)
        if len(temp)!=0:
            test['Past5']= temp
        else:
            test['Past5']=136
        
        test['weighted_avg'] = (3*test['Past1'] + 2*test['Past2'] + test['Past3'] )/6
        
        test['week_diff'] = abs(test['Past1']-test['Past2'])
        
        return test 
    
#Loading the saved model,files.
def load_model():
    onehot = pickle.load(open('onehot', "rb"))
    scaler = pickle.load(open('scaler', "rb"))
    model = pickle.load(open('XGBOOST', "rb"))    
    return onehot,scaler,model

#Function to predict sales in the first module
def predic(test,database):
    
    onehot,scaler,model = load_model()
    
    #print(database)
    prediction = 0
    test = test.reset_index()
    del test['index']
    
    temp1 = fea(database,test)
    
    cat_features=['center_id','meal_id']
    
    encoded = onehot.transform(temp1[cat_features])
    pl1 = pd.DataFrame(encoded.toarray())
    temp1.drop(cat_features,axis=1,inplace=True)
    temp1=pd.concat([temp1,pl1],axis=1)

    for col in ['checkout_price','Past1','Past2','Past4','Past3','Past5','price_diff','weighted_avg','week_diff']:

        y = np.array(temp1[col]) #returns a numpy array 
        y = np.reshape(y,(-1,1))
        y_scaled = scaler.transform(y)
        temp1[col] = y_scaled
    
    #st.write(temp1)
    prediction = model.predict(temp1)

    return prediction



#Main function of Modele one to forecast sales
def Forecast() :
    database = load_database()
    
    COLUMN_NAMES=['id', 'week', 'center_id', 'meal_id', 'checkout_price',
       'emailer_for_promotion', 'homepage_featured','price_diff']
    df = pd.DataFrame(columns=COLUMN_NAMES)
    st.title('Forecasting Store Sales')
    
    st.subheader("This module takes the takes the input and forecast sales for that paticular week.")
    st.write('Fill all the details and click Forecast Sales button for prediction.')
    number = st.number_input('Insert an Id of the product:',format='%d',min_value=1000000,max_value=9999999)
    st.write('The current number is:', number)
    
    week = st.number_input('Week number:',format='%d',min_value=1,max_value=1000)
    st.write('The current week is ', week)
    
    option_center = st.selectbox(
     'Please select your Center ID:',
     (10,11,13,14,17,20,23,24,26,27,29,30,32,34,36,39,41,42,43,50,51,52,53,55,57,58,59,
      61,64,65,66,67,68,72,73,74,75,76,
      77,80,81,83,86,88,89,91,92,93,94,97,99,101,102,104,106,108,109,110,113,124,126,
      129,132,137,139,143,145,146,149,152,153,157,161,162,174,177,186))

    st.write('You selected:', option_center)
    st.caption('If know more information about centers, please check the following checkbox.')
    
    center_check = st.checkbox('Center Information')

    if center_check:
       
        st.write(load_center())
        
    option_meal = st.selectbox('Please select your Meal ID:',
      (1062,1109,1198,1207,1216,1230,1247,1248,1311,1438,1445,1525,1543,1558,1571,1727,
       1754,1770,1778,1803,1847,1878,1885,1902,1962,1971,1993,2104,2126,2139,2290,2304,
       2306,2322,2444,2490,2492,2494,2539,2569,2577,2581,2631,2640,2664,2704,2707,2760,
       2826,2867,2956))

    st.write('You selected:', option_meal)
    st.caption('If know more information about Meal, please check the following checkbox.')
     
    meal_check = st.checkbox('Meal Information')

    if meal_check:
        
         st.write(load_meal())
    checkout = st.number_input('Checkout price of the product',min_value=1.00)
    st.write('The current number is ', checkout)
    
    base = st.number_input('Base price of the product',min_value=1.00)
    st.write('The current number is ', base)
        
    email = st.radio(
     "Did you send E-mail for promotion",
     ('No', 'Yes' ))
    if email =='Yes':
        email = 1
    else:
        email=0
        
    homepage = st.radio(
     "Did you promote the product on website",
     ('No', 'Yes' ))
    if homepage =='Yes':
        home= 1
    else:
        home=0
        
    if st.button('Forecast sales'):
        df = df.append({'id':int(number), 'week':float(week), 'center_id':int(option_center), 'meal_id':int(option_meal), 'checkout_price':int(checkout),
           'emailer_for_promotion':int(email), 'homepage_featured':int(home), 'price_diff': int(base - checkout)}, ignore_index=True)
        
        prediction = predic(df,database)
        st.subheader('Forcasted Sales :',)
        if int(prediction) < 0 :
            st.title('0')
        else:
            st.title(int(prediction))


def load_model_2():
    
    model = pickle.load(open('model_2', "rb"))    
    scaler = pickle.load(open('scaler_2', "rb"))   
    return model,scaler

#Forecasting sales on the previous sales
def Forecast_previous():
    
    COLUMN_NAMES=['week', 'Past1', 'Past2', 'Past3', 'Past4', 'Past5', 'weighted_avg']
    df = pd.DataFrame(columns=COLUMN_NAMES)
    
    st.title('Forecasting Sales')
    
    st.subheader("This module takes previous sales as input and forecast sales for the next week.")
    st.write('Fill all the details and click Forecast Sales button for prediction.')
    st.write('Note - The accuracy is lower, comparitively to the other module.')
    
    number = st.number_input('Week number:',format='%d',min_value=1)
    st.write('The current number is:', number)
    
    number1 = st.number_input('Sales 1 week back:',format='%d',min_value=0,max_value=2000)
    st.write('The current number is:', number1)
    
    number2 = st.number_input('Sales 2 week back:',format='%d',min_value=0,max_value=2000)
    st.write('The current number is:', number2)
    
    number3 = st.number_input('Sales 3 week back:',format='%d',min_value=0,max_value=2000)
    st.write('The current number is:', number3)
    
    number4 = st.number_input('Sales 4 week back:',format='%d',min_value=0,max_value=2000)
    st.write('The current number is:', number4)
    
    number5 = st.number_input('Sales 5 week back:',format='%d',min_value=0,max_value=2000)
    st.write('The current number is:', number5)

    if st.button('Forecast sales'):
        temp = (3*int(number1) +2*int(number2)+int(number3))/6
        df = df.append({'week':int(number), 'Past1':int(number1), 'Past2':int(number2), 'Past3':int(number3), 'Past4':int(number4), 'Past5':int(number5), 'weighted_avg':temp}, ignore_index=True)
        model,scaler = load_model_2()
        
        for col in ['Past1','Past2','Past4','Past3','Past4','Past5','weighted_avg']:

            
            y = np.array(df[col]) #returns a numpy array 
            y = np.reshape(y,(-1,1))
            y_scaled = scaler.transform(y)
            df[col] = y_scaled
        prediction = model.predict(df)
        st.subheader('Forcasted Sales :',)
        if int(prediction) < 0 :
            st.title('0')
        else:
            st.title(int(prediction))
       



st.sidebar.title("What to do")
app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show Instruction", "Forecast Store Sales", "Forecast on previous sales","Show the source code","About"])
 
if app_mode == "Show Instructions":
    st.sidebar.success('Select a operation to perform')
    
elif app_mode == "Forecast Store Sales":
    readme_text.empty()
    Forecast()    
        
elif app_mode == "Show the source code":
    readme_text.empty()
    st.code(get_file_content_as_string("streamlit_app.py"))
    
elif app_mode == "Forecast on previous sales":
    readme_text.empty()
    Forecast_previous()
    
elif app_mode == "About":
    readme_text.empty()
    st.markdown(get_file_content_as_string('about.md'))
