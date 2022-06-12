import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

vector=joblib.load('./vector1')
m_jlib1 = joblib.load('./model_jlib1')

def welcome():
	return 'welcome all'
def prediction(text):
    text=[text]
    test =vector.transform(text)
    pred = m_jlib1.predict(test)
    return pred
	

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	st.title("fakenews Prediction")
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Streamlit fakenews ML App </h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	text_input = st.text_input("Text", "Type Here")
	result =""
	
	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if st.button("Predict"):
		result = prediction(text_input)
	st.success('The output is {}'.format(result))
	
if __name__=='__main__':
	main()
