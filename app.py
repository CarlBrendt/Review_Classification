import tensorflow as tf
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import streamlit as st

@st.cache()
def load_data():
    with open('x_train_for_tokinize.json','r') as file:
        x_train_for_tokinize = json.load(file)
    return x_train_for_tokinize
    
TAG_re = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_re.sub('',text)

def preprocess_text(sen):
    
    sentence = sen.lower()

    # Remove html tags
    sentence = remove_tags(sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'c1_lstm_model_acc_0.843-0.1.0.keras'
    pretrained_lstm_model = tf.keras.models.load_model(model_path)
    return pretrained_lstm_model


with st.spinner('Model is being loaded..'):
   pretrained_lstm_model=load_model()

def get_prediction(text):

	x_train_for_tokinize = load_data()
		
	word_tokenizer = tf.keras.preprocessing.text.Tokenizer()
	word_tokenizer.fit_on_texts(x_train_for_tokinize)
	
	maxlen = 100
	unseen_processed = [preprocess_text(text)]
  
	unseen_tokenized = word_tokenizer.texts_to_sequences(unseen_processed)

	# Pooling instance to have maxlength of 100 tokens
	unseen_padded = tf.keras.preprocessing.sequence.pad_sequences(unseen_tokenized, padding='post', maxlen=maxlen)
	unseen_sentiments = pretrained_lstm_model.predict(unseen_padded)
	rating = np.round(unseen_sentiments*10,1)

	return rating[0][0]

def main():
    st.title('Review Classification')
    
    review_text = st.text_input('Movie Review Text')
    
    result = 0
    
    #button for prediction
    if st.button('Predict rating'):
        result = get_prediction(review_text)
    if result < 6 and result !=0:
        st.error(result)
    else:
        st.success(result)
    if result > 6:
        review = '<p style="font-family:sans-serif; color:Green; font-size: 22px;">Positive Review</p>'
        st.markdown(review, unsafe_allow_html=True)
    elif result <= 6 and result != 0:
        review = '<p style="font-family:sans-serif; color:Red; font-size: 25px;">Negative Review</p>'
        st.markdown(review, unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()