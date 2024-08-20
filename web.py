
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import re
import itertools
import pdfplumber


#model_imports
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#chat_bot_imports
import tempfile
import pdfplumber
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage


# Set up Streamlit configuration
st.set_page_config(page_title="Bank Risk Controller Systems - Made by: Naveen A", layout="wide")

# Load sample data
sample = pd.read_csv('eda_data.csv')
model_data = pd.read_csv('model_data.csv')

def load_pickle(file_name):
    """Load a pickle file and return its content."""
    try:
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"File not found: {file_name}")
        return None
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return None

# Load encoders
encoders = load_pickle('label_encoders.pkl')
if encoders is None:
    st.stop()

# Retrieve classes from encoders
ORGANIZATION_TYPE = encoders['ORGANIZATION_TYPE'].classes_.tolist()
OCCUPATION_TYPE = encoders['OCCUPATION_TYPE'].classes_.tolist()

def get_user_input():
    """Get user input from Streamlit form."""
    st.subheader(":violet[Fill all the fields and press the button below to view **The user Defaulter or Non-defaulter**:]")
    cc1, cc2 = st.columns([2, 2])
    
    with cc1:
        BIRTH_YEAR = st.number_input("Birth Year (YYYY):", min_value=1950, max_value=2024)
        AMT_CREDIT = st.number_input("Credit Amount of loan:")
        AMT_ANNUITY = st.number_input("Loan Annuity:")
        AMT_INCOME_TOTAL = st.number_input("Income of the user:")
        ORGANIZATION_TYPE_input = st.selectbox("Organization Type:", ORGANIZATION_TYPE)
    
    with cc2:
        OCCUPATION_TYPE_input = st.selectbox("Occupation Type:", OCCUPATION_TYPE)
        EXT_SOURCE_2 = st.number_input("Score from External-2 data source:")
        EXT_SOURCE_3 = st.number_input("Score from External-3 data source:")
        REGION_POPULATION_RELATIVE = st.number_input("Population of the region:")
        HOUR_APPR_PROCESS_START = st.number_input("Hour user applied for the loan:")
        EMPLOYMENT_START_YEAR = st.number_input("Employment Start Year:", min_value=1950, max_value=2024)

    user_input_data = {
        'BIRTH_YEAR': BIRTH_YEAR,
        'AMT_CREDIT': AMT_CREDIT,
        'AMT_ANNUITY': AMT_ANNUITY,
        'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
        'ORGANIZATION_TYPE': ORGANIZATION_TYPE_input,
        'OCCUPATION_TYPE': OCCUPATION_TYPE_input,
        'EXT_SOURCE_2': EXT_SOURCE_2,
        'EXT_SOURCE_3': EXT_SOURCE_3,
        'REGION_POPULATION_RELATIVE': REGION_POPULATION_RELATIVE,
        'HOUR_APPR_PROCESS_START': HOUR_APPR_PROCESS_START,
        'EMPLOYMENT_START_YEAR': EMPLOYMENT_START_YEAR
    }
    
    return pd.DataFrame(user_input_data, index=[0])

def load_model():
    """Load the Extra Trees Classifier model."""
    try:
        with open('ET_Classifier_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found: ET_Classifier_model.pkl")
        return None
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

def data_transformation_for_the_model(df):
    """Transform data using pre-loaded encoders."""
    df = df.copy()  # Avoid modifying the original DataFrame
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    return df

def get_response(question, context):
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    # Initialize the model
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    model = HuggingFaceEndpoint(repo_id=repo_id, max_length=50, temperature=0.1, token=hf_token)

    # Define the prompt template
    template = """
    You are an assistant that provides answers to questions based on
    a given context. 

    respond back to the question based on the context. If you can't answer the
    question, reply "I don't know".
    
    if a user asked you question like "can u help me" reply like Yes, "I am a chatbot i can able to answer your questions based on the given document.".
    
    Dont response back like Answer: A legally binding contract between a lender and a borrower outlining the terms and conditions of a loan.
    
    directly start your resonse like  "A legally binding contract between a lender and a borrower outlining the terms and conditions of a loan".

    Be as concise as possible and go straight to the point.

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    # Create the prompt with the context and question
    prompt_text = prompt.format(context=context, question=question)
    
    # Get the model's response
    response = model(prompt_text)
    
    return response

def plot(sample, col, title, pie_colors):
    bar_color = '#7B68EE' 

    plt.figure(figsize=(10, 5))
    value_counts = sample[col].value_counts()

    plt.subplot(121)
    value_counts.plot.pie(
        autopct="%1.0f%%",
        colors=pie_colors[:len(value_counts)],
        startangle=60,
        wedgeprops={"linewidth": 2, "edgecolor": "k"},
        explode=[0.1] * len(value_counts),
        shadow=True
    )
    plt.title(f"Distribution of {title}")
    plt.subplot(122)
    ax = sample[col].value_counts().plot(kind="barh", color=bar_color)
    for i, (value, label) in enumerate(zip(value_counts.values, value_counts.index)):
        ax.text(value, i, f' {value}', weight="bold", fontsize=12, color='black')

    plt.title(f"Count of {title}")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def main():
    """Main function to control Streamlit app flow."""
    global text_chunks  # Declare text_chunks as global for use in retrieve_relevant_chunks
    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("Select options")
        choice = st.radio("Navigation", ["Data", "EDA", "Model", "Multiple prediction", "Chat with PDF"])
        st.info("This project application helps you predict defaulters and non-defaulters")
    
    if choice == "Data":
        st.title(":violet[Welcome to the Bank Risk Controller Systems Prediction App]")   
        st.write('### :violet[Model Performance Metrics]')
        metrics = {
            'Model': ['ExtraTreesClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier'],
            'Accuracy': [1.0000, 0.9999, 0.9971, 0.7661],  
            'Precision': [1.0000, 0.9999, 0.9971, 0.7665],
            'Recall': [1.0000, 0.9999, 0.9971, 0.7661],
            'F1 Score': [1.0000, 0.9999, 0.9971, 0.7660],
            'ROC AUC': [0.9999, 0.9999, 0.9970, 0.7660],
            'Confusion Matrix': [
                '[[161689, 5], [0, 162520]]',
                '[[161675, 19], [0, 162520]]',
                '[[160746, 948], [0, 162520]]',
                '[[120689, 41005], [34833, 127687]]']
        }
        metrics_df = pd.DataFrame(metrics) 
        st.dataframe(metrics_df)
        st.write(":violet[Sample Dataset]")
        st.write(model_data.head(11))
        st.write('## :violet[Created by] \n ### Naveen Anandhan')
        
    if choice == "Model":
        st.title(":violet[Bank Risk Controller Systems App]")
        user_input_data = get_user_input()

        if st.button("Predict"):
            df = data_transformation_for_the_model(user_input_data)
            model = load_model()
            if model is not None:
                prediction = model.predict(df)
                st.success(f'Prediction: {"Defaulter" if prediction[0] == 1 else "Non-defaulter"}')
            
    if choice == "Multiple prediction":
        st.title(":violet[Multiple Users Prediction]")
        
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            input_data = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV Data:", input_data)
            input_data_dropped = input_data.iloc[:, :-1]
            df_transformed = data_transformation_for_the_model(input_data_dropped)
            model = load_model()
            if model is not None:
                predictions = model.predict(df_transformed)
                input_data['Prediction'] = predictions
                st.write("Predictions:", input_data)
    
    if choice == "Chat with PDF":
        
        st.title(":violet[Chat with Your PDF]")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        def extract_text_from_pdf(uploaded_file):
            full_text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text
            return full_text
        
        

        if uploaded_file is not None:
       
            full_text = extract_text_from_pdf(uploaded_file)
            
            st.text_area("PDF Content", full_text, height=100)
            
            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
            chunks = splitter.split_text(full_text)

            # Initialize embeddings and vector store
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            vectorstore = FAISS.from_texts(chunks, embeddings)
            retriever = vectorstore.as_retriever()

            # Initialize session state for chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                    AIMessage(content="Hello, How can I help you?"),
                ]
            
            ai_icon_url = "https://www.stemrobo.com/wp-content/uploads/2023/07/1.png"
            
            # Display conversation
            for message in st.session_state.chat_history:
                
                
                
                if isinstance(message, HumanMessage):
                    st.markdown(f"""
                    <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                        <div style='max-width: 70%; background-color: #778da9; color: white; padding: 10px; border-radius: 20px;'>
                            {message.content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif isinstance(message, AIMessage):
                    st.markdown(f"""
                    <div style='display: flex; align-items: flex-end; margin-bottom: 10px;'>
                        <img src="{ai_icon_url}" alt="AI Icon" style='width: 50px; height: 50px; margin-right: 10px;'>
                        <div style='max-width: 70%; background-color: #85182a; color: white; padding: 10px; border-radius: 20px;'>
                            {message.content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # User input
            user_query = st.chat_input("Type your message here...")
            if user_query:
                # Add user query to chat history
                st.session_state.chat_history.append(HumanMessage(content=user_query))

                # Retrieve relevant context
                relevant_chunks = retriever.get_relevant_documents(user_query)
                context = "\n".join([chunk.page_content for chunk in relevant_chunks])

                # Get AI response
                response = get_response(user_query, context)
                r = str(response)
                #answer = response.split('Answer: ')[1].strip()
                answer = re.sub(r'^Answer:\s*', '', r).strip()

                
                # Display user query and AI response in chat  
                st.markdown(f"""
                <div style='max-width: 70%; background-color: #778da9; color: white; padding: 10px; border-radius: 20px; margin-bottom: 10px; float: right; clear: both;'>
                {user_query}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='display: flex; align-items: flex-end; margin-bottom: 10px;'>
                        <img src="{ai_icon_url}" alt="AI Icon" style='width: 50px; height: 50px; margin-right: 10px;'>
                        <div style='max-width: 70%; background-color: #85182a; color: white; padding: 10px; border-radius: 20px;'>
                            {answer}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Add AI response to chat history
                st.session_state.chat_history.append(AIMessage(content=answer))
                
                
    if choice == "EDA":
        
        col1,col2= st.columns(2)
        
        with col1:
            st.title(":violet[Distribution of Gender in dataset]")
            pie_colors = ["#ff006e", "#ffd60a", '#6a0dad', '#ff4500']
            plot(sample, "CODE_GENDER", "Gender", pie_colors)
            st.markdown(
            '''
            <p style="color: #ffee32; font-size: 30px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">67%</span> Female and 
                        <span style="color: #ef233c; font-size: 40px ">33%</span> Male<br>
                </strong>
            </p>''',
            unsafe_allow_html=True)
        with col2:
            st.title(":violet[Distribution of target]")
            pie_colors = ['#d00000', '#ffd500','#f72585', '#ff4500']
            plot(sample, "TARGET", "Target", pie_colors)
            st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong>
                        1 - defaulter,
                        0 - non-defaulter<br>
                        he/she had late payment more than X days are defaulter.<br>
                        <span style="color: #ef233c; font-size: 40px ">8%</span> Defaulter, <span style="color: #ef233c; font-size: 40px ">92%</span> Non-defaulter
                    </strong>
                </p>
                ''',
                unsafe_allow_html=True)
            
        col1,col2= st.columns(2)
        
        with col1: 
            st.title(":violet[Distribution in Contract types in loan_data] \n - Revolving loan \n - Cash loan")
            
            pie_colors = ['#ff006e', '#d00000','#f72585', '#ff4500']
            plot(sample, "NAME_CONTRACT_TYPE_x", "Loan Type", pie_colors)
            st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">8%</span>
                        of user in dataset are Revolving loan type its a form of credit allows the user to borrower to withdraw,
                        repay, and withdraw again up to a certain limit.
                    </strong>
                </p>
                ''',
                unsafe_allow_html=True)
        with col2:
            st.title(":violet[Distribution of loan type by Gender] \n - Female are more loan taked then male \n - Cash loans is always prefered over Revolving loans by both genders")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(x="NAME_CONTRACT_TYPE_x", hue="CODE_GENDER", data=sample, palette=["#00bbf9", "#f15bb5", "#ee964b"], ax=ax)
            ax.set_facecolor("#020202")
            ax.set_title("Distribution of Contract Type by Gender")
            st.pyplot(plt)
            plt.close()
        
        col1,col2= st.columns(2)
        
        with col1: 
            st.title(":violet[Distribution of Own car]")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            
            pie_colors = ['#ff006e', '#d00000','#f72585', '#ff4500']
            plot(sample, "FLAG_OWN_CAR", "Distribution of own car", pie_colors)
          
            
            st.markdown(
            '''
            <p style="color: #ffee32; font-size: 30px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">34%</span>of users own a car. 
                        <span style="color: #ef233c; font-size: 47px ">66%</span> users didn't own a car <br>
                </strong>
            </p>''',
            unsafe_allow_html=True)
            
        with col2:
            st.title(":violet[Distribution Owning a Car by Gender]")
            fig = plt.figure(figsize=(4, 2))
            ax = plt.subplot(121)

            value_counts = sample[sample["FLAG_OWN_CAR"] == "Y"]["CODE_GENDER"].value_counts()
            pie_colors = ['#ff9999', '#66b3ff', '#99ff99']
            value_counts.plot.pie(
                autopct="%1.0f%%",
                colors=pie_colors[:len(value_counts)],
                startangle=60,
                wedgeprops={"linewidth": 2, "edgecolor": "k"},
                explode=[0.1] * len(value_counts),
                shadow=True,
                ax=ax
            )
            ax.set_title("Distribution Owning a Car by Gender")   
            st.pyplot(plt)
            plt.close()
            
            st.markdown(
            '''
            <p style="color: #ffee32; font-size: 30px; text-align: center;">
                    <strong> Out of own car users
                        <span style="color: #ef233c; font-size: 40px ">55%</span> are Male. 
                        <span style="color: #ef233c; font-size: 40px ">45%</span> are Female.
                </strong>
            </p>''',
            unsafe_allow_html=True)
            
        col1,col2= st.columns(2)
        
        with col1: 
            st.title(":violet[Distribution of owning a house or flat]")
            
            pie_colors = ['#ff006e', '#d00000','#f72585', '#ff4500']
            plot(sample, "FLAG_OWN_REALTY", "Distribution of owning a house or flat", pie_colors)
            st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">72%</span>
                        of users own a flat or house.
                    </strong>
                </p>
                ''',
                unsafe_allow_html=True)
        with col2:
            st.title(":violet[Distribution of owning a house or flat by gender]")
            
            fig = plt.figure(figsize=(5, 3))
            ax = plt.subplot(121)
            value_counts = sample[sample["FLAG_OWN_REALTY"] == "Y"]["CODE_GENDER"].value_counts()
            pie_colors = ['#ff9999', '#66b3ff', '#99ff99']
            value_counts.plot.pie(
                autopct="%1.0f%%",
                colors=pie_colors[:len(value_counts)],
                startangle=60,
                wedgeprops={"linewidth": 2, "edgecolor": "k"},
                explode=[0.1] * len(value_counts),
                shadow=True,
                ax=ax
            )
            ax.set_title("Distribution of owning a house or flat by gender")
            st.pyplot(plt)
            plt.close()
            
            st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong> Out of own flat users
                        <span style="color: #ef233c; font-size: 40px ">69%</span> are female
                        <span style="color: #ef233c; font-size: 40px ">31%</span> are male
                    </strong>
                </p>
                ''',
                unsafe_allow_html=True)
            
        col1,col2= st.columns(2)
        
        with col1:
            st.title(":violet[Distribution of Number of Children by Repayment Status]")
            
            fig = plt.figure(figsize=(5, 5))
            plt.subplot(211)

            sns.countplot(x="CNT_CHILDREN", hue="TARGET", data=sample, palette="pastel")
            plt.legend(loc="upper right", prop={'size': 20})
            plt.title("Distribution of Number of Children by Repayment Status")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
            st.write("- if the user has less childer they most likely to be Non-defaulter")
            
        with col2:
            st.title(":violet[Distribution of Number of Family Members by Repayment Status]")
            
            fig = plt.figure(figsize=(5, 5))
            plt.subplot(211)
            
            sns.countplot(x="CNT_FAM_MEMBERS", hue="TARGET", data=sample, palette="Set2")
            plt.legend(loc="upper right",prop={'size': 18})
            plt.title("Distribution of Number of Family Members by Repayment Status")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
            st.write("""- if the user has less family members there Repayment status is high.\n - if the user has only family member of 2 Repayment Status is very higy comparing to higher family members.
                     """)
        
        st.title(":violet[Distribution of Defaulter and non-Defaulter] \n - Loan type \n- Gender \n - Own car \n- Own house")

        default = sample[sample["TARGET"]==1][[ 'NAME_CONTRACT_TYPE_x', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]
        non_default = sample[sample["TARGET"]==0][[ 'NAME_CONTRACT_TYPE_x', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]

        d_cols = ['NAME_CONTRACT_TYPE_x', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
        d_length = len(d_cols)

        fig = plt.figure(figsize=(16,4))
        for i,j in itertools.zip_longest(d_cols,range(d_length)):
            plt.subplot(1,4,j+1)
            default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism"),startangle = 90,
                                                wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)
            circ = plt.Circle((0,0),.7,color="white")
            plt.gca().add_artist(circ)
            plt.ylabel("")
            plt.title(i+"-Defaulter")
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

        fig = plt.figure(figsize=(16,4))
        for i,j in itertools.zip_longest(d_cols,range(d_length)):
            plt.subplot(1,4,j+1)
            non_default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",3),startangle = 90,
                                                wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)
            circ = plt.Circle((0,0),.7,color="white")
            plt.gca().add_artist(circ)
            plt.ylabel("")
            plt.title(i+"-Repayer")
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        st.write("""              
        - 3% Percentage of Cash Loans has more defaults than Revolving Loans.
        - 10% Percentage of males more defaults than non defaulters. 
        - 8% Percentage of female are more repayers.
        """)
        
        
        st.title(":violet[Comparing summary statistics between defaulters and non - defaulters for loan amounts]")
       
        cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT_x','AMT_ANNUITY_x', 'AMT_GOODS_PRICE_x']
        df = sample.groupby("TARGET")[cols].describe().transpose().reset_index()
        df = df[df["level_1"].isin(['mean', 'std', 'min', 'max'])] 

        df_x = df[["level_0", "level_1", 0]].rename(columns={'level_0': "amount_type", 'level_1': "statistic", 0: "amount"})
        df_x["type"] = "REPAYER"

        df_y = df[["level_0", "level_1", 1]].rename(columns={'level_0': "amount_type", 'level_1': "statistic", 1: "amount"})
        df_y["type"] = "DEFAULTER"

        df_new = pd.concat([df_x, df_y], axis=0)

        stat = df_new["statistic"].unique().tolist()
        length = len(stat)

        plt.figure(figsize=(8, 8))

        for i, j in itertools.zip_longest(stat, range(length)):
            plt.subplot(2, 2, j + 1)
            sns.barplot(x="amount_type", y="amount", hue="type",
                        data=df_new[df_new["statistic"] == i], palette=["g", "r"])
            plt.title(i + " -- Defaulters vs Non-defaulters")
            plt.xticks(rotation=35)
            plt.subplots_adjust(hspace=0.4)
            plt.gca().set_facecolor("lightgrey")

        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        st.write("""              
        # Income of users
        - 1 . Average income of users who default and non-defaulter are almost same.

        - 2 . Standard deviation in income of default is very high compared to non-defaulter.

        - 3 . Default also has maximum income earnings

        # Credit amount of the loan credited , Loan annuity, Amount goods price

        - 1 . Statistics between *credit amounts*, *Loan annuity* and *Amount goods price* given in the data the default and non-defaulter are almost similar.
        """)
       
        st.title(":violet[Average Income,credit,annuity & goods_price by gender]")
        
        df1 = sample.groupby("CODE_GENDER")[cols].mean().transpose().reset_index()

        df_f = df1[["index", "F"]].rename(columns={'index': "amt_type", 'F': "amount"})
        df_f["gender"] = "FEMALE"

        df_m = df1[["index", "M"]].rename(columns={'index': "amt_type", 'M': "amount"})
        df_m["gender"] = "MALE"

        df_xna = df1[["index", "XNA"]].rename(columns={'index': "amt_type", 'XNA': "amount"})
        df_xna["gender"] = "XNA"

        df_gen = pd.concat([df_m, df_f, df_xna], axis=0)


        plt.figure(figsize=(6, 3))
        ax = sns.barplot(x="amt_type", y="amount", data=df_gen, hue="gender", palette="Set1")
        plt.title("Average Income, Credit, Annuity & Goods Price by Gender")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
       
        st.title(":violet[Distribution of Suite type]\n - NAME_TYPE_SUITE - Who was accompanying user when he was applying for the loan.")
        
        col1,col2= st.columns(2)
        with col1:
            plt.figure(figsize=(10, 3))
            plt.subplot(121)
            sns.countplot(y=sample["NAME_TYPE_SUITE_x"],
                        palette="Set2",
                        order=sample["NAME_TYPE_SUITE_x"].value_counts().index[:5])
            plt.title("Distribution of Suite Type")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        with col2:
            plt.figure(figsize=(10, 3))
            plt.subplot(122)
            sns.countplot(y=sample["NAME_TYPE_SUITE_x"],
                        hue=sample["CODE_GENDER"],
                        palette="Set2",
                        order=sample["NAME_TYPE_SUITE_x"].value_counts().index[:5])
            plt.ylabel("")
            plt.title("Distribution of Suite Type by Gender")
            plt.legend(loc="lower right", prop={'size': 10})
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
        st.title(":violet[Distribution of client income type]\n - NAME_INCOME_TYPE Clients income type (businessman, working, maternity leave...)")
        
        col1,col2= st.columns(2)
        with col1:
            plt.figure(figsize=(10, 3))
            plt.subplot(121)
            sns.countplot(y=sample["NAME_INCOME_TYPE"],
                        palette="Set2",
                        order=sample["NAME_INCOME_TYPE"].value_counts().index[:4])
            plt.title("Distribution of Client Income Type")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        with col2:
            plt.figure(figsize=(10, 3))
            plt.subplot(122)
            sns.countplot(y=sample["NAME_INCOME_TYPE"],
                        hue=sample["CODE_GENDER"],
                        palette="Set2",
                        order=sample["NAME_INCOME_TYPE"].value_counts().index[:4])
            plt.ylabel("")
            plt.title("Distribution of Client Income Type by Gender")
            plt.legend(loc="lower right", prop={'size': 10})
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
        
         
        st.title(":violet[Distribution of Education type by loan repayment status]\n - NAME_EDUCATION_TYPE Level of highest education the user achieved..")
        col1,col2= st.columns(2)
        with col1:
            # Plot for Repayers
            plt.figure(figsize=(10, 3))
            plt.subplot(121)
            sample[sample["TARGET"]==0]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(
                fontsize=12,
                autopct="%1.0f%%",
                colors=sns.color_palette("inferno"),
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
                shadow=True
            )
            plt.gca().add_artist(plt.Circle((0, 0), .7, color="white"))  # Add a white circle to create a donut-like effect
            plt.title("Distribution of Education Type for Repayers", color="b")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        
        with col2:
            # Plot for Defaulters
            plt.figure(figsize=(10, 3))
            plt.subplot(122)
            sample[sample["TARGET"]==1]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(
                fontsize=12,
                autopct="%1.0f%%",
                colors=sns.color_palette("Set2"),
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
                shadow=True
            )
            plt.gca().add_artist(plt.Circle((0, 0), .7, color="white"))  # Add a white circle to create a donut-like effect
            plt.title("Distribution of Education Type for Defaulters", color="b")
            plt.ylabel("")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">8%</span> perentage of users with higher education are less defaulter compared to user non-defaulter.
                    </strong>
                </p>
                ''', unsafe_allow_html=True)
        
        st.title(":violet[Distribution of Education type by loan repayment status]")
        
        edu = sample.groupby(['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index().sort_values(by='AMT_INCOME_TOTAL', ascending=False)
        # Create the bar plot
        fig = plt.figure(figsize=(10, 5))
        ax = sns.barplot(x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', data=edu, hue='NAME_EDUCATION_TYPE', palette="seismic")
        ax.set_facecolor("k")
        plt.title("Average Earnings by Different Professions and Education Types")
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        st.title(":violet[Distribution normalized population of region where client lives by loan repayment status]\n - REGION_POPULATION_RELATIVE - Normalized population of region where client lives (higher number means the client lives in more populated region).")
        fig = plt.figure(figsize=(10,5))

        plt.subplot(121)
        sns.violinplot(y=sample[sample["TARGET"]==0]["REGION_POPULATION_RELATIVE"],
                    x=sample[sample["TARGET"]==0]["NAME_CONTRACT_TYPE_x"],
                    palette="Set1")
        plt.title("Distribution of Region Population for Non-Default Loans", color="b")

        plt.subplot(122)
        sns.violinplot(y=sample[sample["TARGET"]==1]["REGION_POPULATION_RELATIVE"],
                    x=sample[sample["TARGET"]==1]["NAME_CONTRACT_TYPE_x"],
                    palette="Set1")
        plt.title("Distribution of Region Population for Default Loans", color="b")

        plt.subplots_adjust(wspace=.2)
        fig.set_facecolor("lightgrey")
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong> Point to infer from the graph
                    \n- In High population density regions people are less likely to default on loans.
                    </strong>
                </p>
                ''', unsafe_allow_html=True)
if __name__ == "__main__":
    main()
