import streamlit as st
import pandas as pd
import joblib
# Function to load the model
@st.cache_data
def load_model():
    with open('insurance_model', 'rb') as file:
        loaded_model = joblib.load(file)
    return loaded_model

# Load your model
loaded_model = load_model()


# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Medical Insurance Cost Prediction Web App')
    # Index(['age', 'sex', 'bmi', 'children', 'smoker', 'region'], dtype='object')
    # User inputs
    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    bmi = st.number_input('BMI', min_value=10, max_value=60, value=25)
    children = st.number_input('Children', min_value=0, max_value=10, value=0)
    smoker = st.selectbox('Smoker', ['Yes', 'No'])
    region = st.selectbox('Region', ['Southwest', 'Southeast', 'Northwest', 'Northeast'])
    
    


    user_inputs = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    # df['sex'] = df['sex'].replace({'male': 0, 'female': 1})
    # df['smoker'] = df['smoker'].replace({'no': 0, 'yes': 1})
    # df['region'] = df['region'].replace({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})
    
    user_inputs['sex'] = 0 if user_inputs['sex'] == 'Male' else 1
    user_inputs['smoker'] = 0 if user_inputs['smoker'] == 'No' else 1
    user_inputs['region'] = 0 if user_inputs['region'] == 'Southwest' else 1 if user_inputs['region'] == 'Southeast' else 2 if user_inputs['region'] == 'Northwest' else 3
    
    if st.button('Predict'):
        prediction = loaded_model.predict(pd.DataFrame(user_inputs, index=[0]))
        st.markdown(f'**The predicted Medical Insurance cost is: {prediction[0]:,.2f}**')  # Display prediction with bold
        
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            # You can include more details about the prediction
            # For example, display the parameters of the loaded model
            st.json(loaded_model.get_params())
            st.write('Model used: Random Forest Regressor')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'insurance_model.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="insurance_model.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('Data')
    # Add a button to download your dataset
    data_path = 'insurance.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="insurance.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Medical-Insurance-Cost-Prediction)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is created to predict the medical insurance cost based on the user inputs such as age, sex, BMI, children, smoker and region. The model used in this web app is a Random Forest Regressor model. The model is trained on the Medical Insurance dataset from Kaggle. The dataset contains 1338 rows and 7 columns. The model is trained to predict the medical insurance cost based on the user inputs. The model is deployed using Streamlit web app.')
    
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Medical-Insurance-Cost-Prediction)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
