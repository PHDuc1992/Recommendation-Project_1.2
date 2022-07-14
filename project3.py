from operator import sub
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import json

# 1. Read data
# url
url_product = 'https://drive.google.com/file/d/1ZabCyBXKPdNWK6RLg28MCa2pjW_oEmdl/view?usp=sharing'
url_review = 'https://drive.google.com/file/d/1byzbt7l36qQoCTFdwbXfVSystmSIyYN-/view?usp=sharing'
url_data_cleand = 'https://drive.google.com/file/d/10t94lNbhv0lqkijtOw8Xosc7FRSo8b25/view?usp=sharing'
url_user_recom_result = 'https://drive.google.com/file/d/1O3_f8hq0kzyXNC1bJfPBHWzWJpdBpwfK/view?usp=sharing'

# function read data
@st.cache
def read_file_from_ggdr(url):
    file_id = url.split('/')[-2]
    dwn_url = 'https://drive.google.com/uc?id=' + file_id
    data = pd.read_csv(dwn_url)
    return data

# 01. Content based filtering
@st.cache
def get_content_based_recommendation(item_name, n):
    df_item = cosine_similarities_recommend.loc[cosine_similarities_recommend['name'] == item_name, :]
    df_item.sort_values(by=['sim_score'], ascending=False, inplace=True)
    df_id = df_item[['item_id_rec']].head(n)
    result = df_id.merge(df_product, left_on='item_id_rec', right_on='item_id')
    return result

#2. Collabratiove filtering
@st.cache
def get_user_recommendation(customer_id, n):
    df_user = user_recommendation.loc[user_recommendation['customer_id'] == customer_id,:]
    df_user.sort_values(by=['rating_pred'], ascending=False, inplace=True)
    result = df_user.head(n)
    return result

#--------------
# GUI
# st.title("Data Science Project")
st.markdown("<h1 style='text-align: center; color: Red;'>Recommendation System</h1>", unsafe_allow_html=True)
# st.markdown("## **Recommendation System**")

menu = ['0. Mục tiêu kinh doanh', '1. Khám phá dữ liệu', '2. Đề xuất dựa trên nội dung', '3. Đề xuất dựa trên đánh giá sản phẩm']

choice = st.sidebar.radio('Danh mục', menu)
if choice == '0. Mục tiêu kinh doanh':
    st.markdown("<h3 style='text-align: left; color: Blue;'>0. Mục tiêu kinh doanh</h3>", unsafe_allow_html=True)
    st.image('tiki.JPG')
    st.write("""
        - Tiki là một hệ sinh thái thương mại “all in one”, trong đó có tiki.vn, là một website thương mại điện tử đứng top 2 của Việt Nam, top 6 khu vực Đông Nam Á. Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.
        """)
    st.write("""
        - Công ty chưa có hệ thống Recommendation System và mục tiêu là có thể xây dựng được hệ thống này giúp đề xuất và gợi ý cho người dùng/ khách hàng'
    """)

elif choice == '1. Khám phá dữ liệu':
    # read data
    data_product = read_file_from_ggdr(url_product)
    data_review = read_file_from_ggdr(url_review)
    # header
    st.markdown("<h3 style='text-align: left; color: Blue;'>1. Khám phá dữ liệu</h3>", unsafe_allow_html=True)

    # body
    st.write("""
        - Dữ liệu được cung cấp sẵn gồm có các tập tin: ProductRaw.csv, ReviewRaw.csv chứa thông tin sản phẩm, review và rating cho các sản phẩm thuộc các nhóm hàng hóa như Mobile_Tablet, TV_Audio, Laptop, Camera, Accessory.
        """)

    st.write('##### 1. Product Rawdata')
    st.dataframe(data_product.head(3))

    st.write('##### 2. Review Rawdata')
    st.dataframe(data_review.head(3))

    st.write('##### 3. Visualization Product Rawdata')
    st.image('01.thietbiso.JPG')
    st.image('02.hangquocte.JPG')
    st.image('03.laptop.JPG')
    st.image('04.mayanh.JPG')
    st.image('05.oto.JPG')
    st.image('06.dienthoai.JPG')
    st.image('07.nhacua.JPG')
    st.image('08.dongho.JPG')
    st.image('09.dienthoai.JPG')



elif choice == '2. Đề xuất dựa trên nội dung':
    # read data
    data_cleaned = read_file_from_ggdr(url_data_cleand)
    df1 = data_cleaned
    cosine_similarities = pd.read_csv('cosine_similarities_10product_v2.csv', index_col=0)
    df_product = data_cleaned[['item_id', 'name', 'rating', 'price', 'brand', 'image', 'group1']]
    cosine_similarities_recommend = cosine_similarities.merge(df_product, left_on='item_id', right_on='item_id')

    # header
    st.markdown("<h3 style='text-align: left; color: Blue;'>2. Đề xuất dựa trên nội dung</h3>", unsafe_allow_html=True)

    # choose bar
    st.sidebar.markdown('***Chọn thông tin***')
    item_name = st.sidebar.selectbox('Tên sản phẩm', df1['name'])
    lst_num = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    items_num = st.sidebar.selectbox('Số sản phẩm muốn đề xuất', lst_num)
    submit_button = st.sidebar.button(label='Summit')

    # result
    if submit_button:

        # display item name choosen:
        st.markdown('***Sản phẩm***')

        st.write('**%s**'%item_name)
        idx_prd = df1.index[df1['name'] == item_name].tolist()[0]
        col1, col2 = st.columns(2)
        col1.image(str(df1.loc[idx_prd,'image']))
        col2.write('Category: %s'%(df1['group1'].iloc[idx_prd]))
        col2.write('Brand: %s'%(df1['brand'].iloc[idx_prd]))
        col2.write('Price: %s'%(df1['price'].iloc[idx_prd]))
        col2.write('Rating: %s'%(df1['rating'].iloc[idx_prd]))    
        
        # result
        st.write('***Top %s***'%items_num, '***sản phẩm tương tự***')

        results = get_content_based_recommendation(item_name, items_num)
        for i in range(0,results.shape[0]):
            st.write('**%s**'%(results['name'].iloc[i]))
            col1, col2 = st.columns(2)
            col1.image(str(results['image'].iloc[i]))
            col2.write('Category: %s'%(results['group1'].iloc[i]))
            col2.write('Brand: %s'%(results['brand'].iloc[i]))
            col2.write('Price: %s'%(results['price'].iloc[i]))
            col2.write('Rating: %s'%(results['rating'].iloc[i]))

elif choice == '3. Đề xuất dựa trên đánh giá sản phẩm':
    # read data
    user_recommendation_result = read_file_from_ggdr(url_user_recom_result)
    data_cleaned = read_file_from_ggdr(url_data_cleand)
    df_product = data_cleaned[['item_id', 'name', 'rating', 'price', 'brand', 'image', 'group1']]
    user_recommendation = user_recommendation_result.merge(df_product, left_on='product_id', right_on='item_id')

    # customers list
    customers_list = user_recommendation_result[['customer_id']]
    customers_list = customers_list.drop_duplicates()

    # header
    st.markdown("<h3 style='text-align: left; color: Blue;'>2. Đề xuất dựa trên đánh giá sản phẩm</h3>", unsafe_allow_html=True)

    # choose bar
    st.sidebar.markdown('***Chọn thông tin***')
    customer_id = st.sidebar.selectbox('ID người dùng', customers_list['customer_id'])
    lst_num = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    items_num = st.sidebar.selectbox('Số sản phẩm muốn đề xuất', lst_num)
    submit_button = st.sidebar.button(label='Summit')

    # result
    if submit_button:

        # display item name choosen:
        st.markdown('***Mã khách hàng***')
        st.write('**%s**'%customer_id)  
        
        # result
        st.write('***Top %s***'%items_num, '***sản phẩm đề xuất***')
        results = get_user_recommendation(customer_id, items_num)
        for i in range(0,results.shape[0]):
            st.write('**%s**'%(results['name'].iloc[i]))
            col1, col2 = st.columns(2)
            col1.image(str(results['image'].iloc[i]))
            col2.write('Category: %s'%(results['group1'].iloc[i]))
            col2.write('Brand: %s'%(results['brand'].iloc[i]))
            col2.write('Price: %s'%(results['price'].iloc[i]))
            col2.write('Rating: %s'%(results['rating'].iloc[i]))
    



