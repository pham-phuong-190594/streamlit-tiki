import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle
import re


# Source Code
raw=pd.read_csv('ProductRaw.csv')
content=pd.read_csv('content.csv')
similar_product=pd.read_csv('similar_product.csv')
cosine_similarities=pickle.load(open('similarity.pkl','rb'))
review=pd.read_csv('ReviewRaw.csv')

def recommend(product):
    product_index=content[content['name']==product].index[0]
    # print("Product Choice:",product)
    # print("Product ID:",content.iloc[product_index].item_id)
    # print("Product Info:")
    # print(content.iloc[product_index].product_content)
    # Display Product Image
    # st.image("https://tiki.vn/"+content.iloc[product_index].url[8:])
    # print("Product Image:",content.iloc[product_index].image)
    distances=cosine_similarities[product_index]
    product_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x: x[1])[:6]
    recommended_product_names = []
    recommended_product_links = []
    recommended_product_posters = []
    recommended_product_price = []
    recommended_product_list_price = []
    recommended_product_description = []
    #print("Top 5 similar products are:")
    for i in product_list:
        product_id=content.iloc[i[0]].item_id
        recommended_product_links.append("https://tiki.vn/"+content.iloc[i[0]].url[8:])
        recommended_product_posters.append(content['image'].iloc[i[0]])
        recommended_product_names.append(content['name'].iloc[i[0]])
        recommended_product_price.append(content['price'].iloc[i[0]])
        recommended_product_list_price.append(content['list_price'].iloc[i[0]])
        recommended_product_description.append("\n"+content['description'].iloc[i[0]])
        #print("\t",content['name'].iloc[i[0]],"(Product ID:",content.iloc[i[0]].item_id,")")
    return recommended_product_links,recommended_product_names, recommended_product_posters,recommended_product_price,recommended_product_list_price,recommended_product_description
def GUI_main(i):
    #col1, col2 = st.columns([3,3])
    #with col1:
    
    #with col2:
    st.subheader(recommended_product_names[i])
    
    col1, col2= st.columns([4,2])
    with col1:
        st.image(recommended_product_posters[i],use_column_width='always')
    with col2:
        st.write("")
        st.metric(label="Original Price", value=recommended_product_list_price[i])
        if recommended_product_list_price[i]>0:
            discount=-(recommended_product_list_price[i]-recommended_product_price[i])/recommended_product_list_price[i]*100
            string=str(round(discount,0))+"%"
            st.metric(label="Discounted Price", value=recommended_product_price[i],delta=string)
        st.write(f'Reference Link:{recommended_product_links[i]}')

def GUI_recommend(i):
    col1, col2 = st.columns([2,4])
    with col1:
        st.image(recommended_product_posters[i],use_column_width='always')
    with col2:
        st.subheader(recommended_product_names[i])
        st.write(f'Reference Link:{recommended_product_links[i]}')
    col1, col2, col3= st.columns([2,2,2])
    with col2:
        st.metric(label="Original Price", value=recommended_product_list_price[i])
    with col3:
        if recommended_product_list_price[i]>0:
            discount=-(recommended_product_list_price[i]-recommended_product_price[i])/recommended_product_list_price[i]*100
            string=str(round(discount,0))+"%"
            st.metric(label="Discounted Price", value=recommended_product_price[i],delta=string)
    
#--------------
# GUI
st.title('Tiki System Recommendation')


#_______________________
menu = ['Bussiness Objective','Content Recommendation Model','Content Recommendation Application']
choice = st.sidebar.selectbox('Menu',menu)
if choice == 'Bussiness Objective':
    st.image('logo-tiki.png')
    st.subheader('Bussiness Objective/Problem')
    st.write('''
    ##### Tiki is an "all in one" commercial ecosystem, in there is tiki.vn, which is a standing e-commerce website Top 2 in Vietnam, top 6 in Southeast Asia.

On tiki.vn website, many advanced support utilities have been deployed high user experience and they want to build many more conveniences.

Assuming Tiki has not implemented Recommender System and you are required to implement this system. What you will do?
    ''')

    st.subheader('Solution Recommendation')
    st.write('''
    ##### Based on the above requirements, we need to build Recommendation System on tiki.vn to give product suggestions to users/customers.

There are 2 types of model we can use for tiki.vn:
- Content based filtering
- Collaborative filtering
    ''')
    st.image('filtering.png')

elif choice == 'Content Recommendation Model':

    st.subheader('Content Recommend System')
    st.caption('''
    We will build Content Recommend System base on Product Information such as Product Name, Product Category & Product Description. 
    Content Recommend System Application:
    - If customers choose a specific product, Content Recommend System will suggest 5 similar products.

    ''')
    st.write('''
    ### 1. Data Understanding & Data Preparation:
    ''')
    st.write('##### Read Product Raw Data')
    st.dataframe(raw.head())
    st.dataframe(raw.tail())
    st.text(raw.shape)

    st.write('##### Product Raw Data after cleaning process')
    # st.dataframe(content.head())

    st.write('''
    ### 2. Data Preparation:
    ''')
    st.write('##### Data with select features to build Content Recommend System')
    st.dataframe(content.head())

    st.write('''
    ### 3. Final Outcome from Model:
    ''')
    st.write('##### Top 5 similar products for all products')
    st.dataframe(similar_product.head())


elif choice == 'Content Recommendation Application':
    st.header("Product Recommendation with Content Recommend System")
    product_list = content['name'].values
    selected_product = st.selectbox("Type or select a product from the dropdown",product_list)
    recommended_product_links,recommended_product_names, recommended_product_posters,recommended_product_price,recommended_product_list_price,recommended_product_description = recommend(selected_product)
    selection = st.selectbox("Select Option:", ("Product Information", "Product Review", "Similar Products"))
    if st.button('Submit'):
        if selection=='Product Information':
            st.write("*"*100)
            st.subheader("Product Information")
            st.write("*"*100)
            GUI_main(0)
            value=recommended_product_description[0].split("\n")
            for text in value[1:-1]:
                if text==text.upper() and text in ["THÔNG TIN CHI TIẾT","MÔ TẢ SẢN PHẨM"]:
                    if bool(re.search(r'\d',text))==False:
                        st.write(f'#### **{text}**')
                else:
                    st.write(f"- {text}")
            #info_df=pd.DataFrame(value,columns=['Product_Info'])
            #st.table(info_df.loc[1:])
    
        if selection=='Product Review':
            st.write("*"*100)
            st.subheader("Product Review")
            st.write("*"*100)
            st.subheader(selected_product)
            st.image(content[content['name']==selected_product]['image'].values[0],use_column_width='always')
            pro_id=content[content['name']==selected_product]['item_id'].values[0]
            if pro_id in review['product_id'].unique().tolist():
                st.write("##### Reviews with detail content:")
                review['content']=review['content'].fillna("")
                review['rating_new']=review['rating'].apply(lambda x: str(x)+"⭐")
                product_review=review[(review['product_id']==pro_id)&(review['content']!="")].sort_values('rating',ascending=False).reset_index()
                product_review=product_review[['name','rating_new','title','content']]
                st.table(product_review)
            else:
                st.write('This product have no reviews.') 
    
        if selection=='Similar Products':
            st.write("*"*100)
            st.subheader("Top 5 similar products")
            st.write("*"*100)
            st.write("-"*100)
            for i in[1,2,3,4,5]:
                st.write(f'Recommend Product {i}')
                GUI_recommend(i)
                st.write("-"*100)