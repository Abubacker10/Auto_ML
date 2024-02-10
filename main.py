from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import pickle
import streamlit as st
from ydata_profiling import ProfileReport
import matplotlib.backends.backend_tkagg
matplotlib.use('tkagg')

df = pd.read_csv('data.csv')
def profiling():
    st.title('Explore The Data & Choose the best Target!')
    pr = ProfileReport(df)
    st_profile_report(pr)

def model():
    col = df.columns
    target = st.selectbox('Select The Target Variable',options=col)
    if st.button('Regression'):
        from pycaret.regression import setup,compare_models,pull
        reg = setup(data=df,target = target)
        bst = compare_models()
        st.write(bst)
        st.write(pull())
        return bst
    elif st.button('Classification'):
        from pycaret.classification import setup,compare_models,pull
        clf = setup(data=df,target=target)
        bst = compare_models()
        st.write(bst)
        st.write(pull())
        return bst


def download(mod):
    st.download_button(
        "Download Model",
        data=pickle.dumps(mod),
        file_name="model.pkl"
    )

def layout():
    st.set_page_config('Auto_ML',layout='wide')
    st.title('Automated ML')
    st.write('Making the Pipeline of ML in Minutes not for hours')
    st.divider()
    option = st.radio('Choose the Pipeline', options=['Uploading...', 'Profiling...', 'Modelling...', 'Downloading...'],horizontal=False)
    if option == 'Uploading...':
        st.title('Upload the Data (.csv)')
        path = st.file_uploader('insert the file', type='csv',key='uploaded_file')
        if path:
            df = pd.read_csv(path,index_col=None)
            df.to_csv('data.csv',index=None)

    if option == 'Profiling...':
        profiling()
    if option=='Modelling...':
        with st.spinner('Modelling the Algorithm..........'):
            mod = model()
            with open('model.pkl','wb') as f:
                pickle.dump(mod,f)
            if mod:
                st.success('Hurray! Best Model is trained out, You can download it by clicking downloading........')
    if option=='Downloading...':
        with st.spinner('Finding....'):
            with open('model.pkl','rb') as f:
                mod = pickle.load(f)
                download(mod)
    st.info(
        """Description: \n This app provide the services for building more accurate and efficient ML model for your dataset in a shorter amount of time \n One beautiful part is, i.e, it just done it without any coding....... """)
    st.info("""Supports : Regression and Classification """)

def main():
    layout()

if __name__ == "__main__":
    main()
