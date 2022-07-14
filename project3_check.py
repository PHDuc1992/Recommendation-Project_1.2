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
url_product = 'https://drive.google.com/file/d/1F5-ebalibG3SH2b7J4yIW00fzaytpE-C/view?usp=sharing'
url_review = 'https://drive.google.com/file/d/1IsYaggVFJpv3MXPVuEygiY3wqV9nd_I0/view?usp=sharing'
url_data_cleand = 'https://drive.google.com/file/d/1PQINiOsojbGjIaBV22-hbHQ_lo3fT_gQ/view?usp=sharing'
url_user_recom_result = 'https://drive.google.com/file/d/1MF4LkOPSwYpZPPq-AE4q4EZ4zYAR3a4r/view?usp=sharing'
url_product_recom_result = 'https://drive.google.com/file/d/1j7d0-sKaMZJj1sws6MPhzdYBdrxzvZO2/view?usp=sharing'
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

menu = ['0. Giới thiệu dự án', '1. Mục tiêu kinh doanh', '2. Khám phá dữ liệu', '3. Đề xuất dựa trên nội dung', '4. Đề xuất dựa trên đánh giá sản phẩm']

choice = st.sidebar.radio('Danh mục', menu)
if choice == '0. Giới thiệu dự án':
    st.write("""#### 1. Khái niệm""")
    st.write("""Recommender system là hệ thống nhằm đề xuất các item có liên quan cho người dùng được sử dụng trong nhiều lĩnh vực: tạo danh sách phát nhạc/video cho các dịch vụ như Netflix, YouTube & Spotify, đề xuất sản phẩm cho các dịch vụ như Amazon V.v""")
    st.write("""#### 2. Hướng tiếp cận""")
    st.write("""**The Collaborative filtering** Collaborative Filtering là một phương pháp gợi ý sản phẩm dựa trên các hành vi của các users khác (collaborative) cùng trên một item để suy ra mức độ quan tâm (filtering) của một user lên sản phẩm""")
    st.write("""**The Content-based filtering** Content-based filtering là một phương pháp gợi ý sản phẩm đánh giá đặc tính của items được recommended. Cách tiếp cận này yêu cầu việc sắp xếp các items vào từng nhóm hoặc đi tìm các đặc trưng của từng item""")
    col1, col2 = st.columns(2)
    col1.image("collaborative-filtering.png")
    col2.image("content-based-filtering-01.png")
    st.write("#### 3. Người thực hiện")
    st.write("""Dự án hoàn thành bởi  ***Phạm Hoài Đức + Trúc Vân *** on July, 2022.""")
    agree = st.checkbox('Đã đọc')
    if agree:
        st.success("""### Xin cảm ơn ! Cùng khám phá nào""")
        st.balloons()
if choice == '1. Mục tiêu kinh doanh':
    st.markdown("<h3 style='text-align: left; color: Blue;'>1. Mục tiêu kinh doanh</h3>", unsafe_allow_html=True)
    st.image('tiki.JPG')
    st.write("""
        - Tiki là một hệ sinh thái thương mại “all in one”, trong đó có tiki.vn, là một website thương mại điện tử đứng top 2 của Việt Nam, top 6 khu vực Đông Nam Á. Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.
        """)
    st.write("""**Sàn thương mại điện tử uy tín nhất Việt Nam** Với 95% khách hàng hài lòng khi trải nghiệm mua sắm trên Tiki, do đó, lượng khách hàng trung thành sẵn sàng mua hàng trên trên Tiki cũng rất cao.""")
    st.write("""**Chi phí kinh doanh cạnh tranh nhất thị trường** Phí thanh toán thấp nhất thị trường và phí chiết khấu cực cạnh tranh.""")
    st.write("""**Lượng truy cập khủng** Trung bình mỗi tháng, sàn TMĐT Tiki có khoảng 100 triệu lượt truy cập vào sàn.""")
    st.write("""**Tỷ lệ trả hàng thấp** Tỷ lệ trả hàng của Tiki cực thấp so với các thương mại điện tử khác trên thị trường, chỉ dưới 0,6%.""")
    st.video('https://youtu.be/04KVix0i-no')
    st.write("""###### => Giả sử: Công ty chưa có hệ thống Recommendation System và mục tiêu là có thể xây dựng được hệ thống này giúp đề xuất và gợi ý sản phẩm cho người dùng/ khách hàng'
    """)

elif choice == '2. Khám phá dữ liệu':
    # read data
    data_product = read_file_from_ggdr(url_product)
    #data_product = pd.read_csv("ProductRaw.csv")
    data_review = read_file_from_ggdr(url_review)
    #data_review = pd.read_csv("ReviewRaw.csv")
    # header
    st.markdown("<h3 style='text-align: left; color: Blue;'>2. Khám phá dữ liệu</h3>", unsafe_allow_html=True)

    # body
    st.write("""
        - Dữ liệu được cung cấp sẵn gồm có các tập tin: ProductRaw.csv, ReviewRaw.csv chứa thông tin sản phẩm, review và rating cho các sản phẩm thuộc các nhóm hàng hóa như Mobile_Tablet, TV_Audio, Laptop, Camera, Accessory.
        """)

    st.write('##### 1. Product Rawdata')
    st.dataframe(data_product.head(3))

    st.write('##### 2. Review Rawdata')
    st.dataframe(data_review.head(3))

    st.write('##### 3. Visualization Rawdata')
    
    st.write("""##### About Product Rawdata""")
    st.image('Product_Price.png')
    st.write("""> ###### Comments
                >
                > - Vùng giá rộng 7k <-> 62.690K
                > - Giá tập trung chủ yếu <3.000K   
                > - Price có nhiều giá trị outliers.""")
    st.write("""""")
    st.image('Product_Brand.png')
    st.write("""> ###### Comments
            > - Thương hiệu Samsung có số lượng mã hàng nhiều nhất(Nếu bỏ qua OEM) các số lượng mã hàng khác giảm dần.""")
    st.image('Product_Brand_Price.png')
    st.write("""> ###### Comments
            > - Hitachi có giá trung bình sản phẩm cao nhất sau đó tới Surface, Bosch.""")
    st.image('Product_Rating.png')
    st.write("""> ###### Comments
                >
                > - Rating trung chủ yếu 4-5
                > - Rating có nhiều giá trị outliers   
                > - Nếu KH ko rating 4-5(tốt) sẽ có xu hướng rating 0(xấu).""")
    st.write("""""")

    st.write("""##### About Review Rawdata""")
    st.image('Review_Rating.png')
    st.write("""> ###### Comments
                >
                > - Rating chủ yếu 5   
                > - Rating có giá trị outliers.""")
    st.write("""""")
    st.image('Review_Customer.png')
    st.write("""> ###### Comments
            > - KH có lượng Review lớn nhất là 50""")
    st.image('Review_Brand.png')
    st.write("""> ###### Comments
            > - Thương hiệu Logitech có lượng review nhiều nhất. Các thương hiệu khác giảm dần""")

    st.write("#### 4. Build model")
    st.write("#### 5. Evaluation")
    evaluation = pd.read_csv("evaluation.csv")
    st.code("Test RMSE: "+ str(round(evaluation.iloc[1]['test_rmse'],2)))
    st.code("Test MAE: "+ str(round(evaluation.iloc[1]['test_mae'],2)))
    st.code("Fit Time: "+ str(evaluation.iloc[1]['fit_time']))
    st.code("Test Time: "+ str(evaluation.iloc[1]['test_time']))

    st.write("#### 6. Summary")
    st.info("##### This model is good enough to build Recommender System for Tiki products.")






elif choice == '3. Đề xuất dựa trên nội dung':
    # read data
    data_cleaned = read_file_from_ggdr(url_data_cleand)
    #data_cleaned = pd.read_csv("product_data_processed.csv")
    df1 = data_cleaned
    #cosine_similarities = pd.read_csv('cosine_similarities_10product_v2.csv', index_col=0)
    cosine_similarities = read_file_from_ggdr(url_product_recom_result)
    cosine_similarities = pd.read_csv('CB_new_v2.csv', index_col=0)
    df_product = data_cleaned[['item_id', 'name', 'rating', 'price', 'brand', 'image', 'group1']]
    cosine_similarities_recommend = cosine_similarities.merge(df_product, left_on='item_id', right_on='item_id')

    # header
    st.markdown("<h3 style='text-align: left; color: Blue;'>2. Đề xuất dựa trên nội dung</h3>", unsafe_allow_html=True)

    # choose bar
    st.markdown('***Chọn thông tin***')
    item_name = st.selectbox('Tên sản phẩm', df1['name'])
    lst_num = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    items_num = st.selectbox('Số sản phẩm muốn đề xuất', lst_num)
    submit_button = st.button(label='Summit')

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

elif choice == '4. Đề xuất dựa trên đánh giá sản phẩm':
    # read data
    user_recommendation_result = read_file_from_ggdr(url_user_recom_result)
    data_cleaned = read_file_from_ggdr(url_data_cleand)
    #user_recommendation_result = pd.read_csv("ALS_user_recs_df_v2.csv")
    #data_cleaned = pd.read_csv("product_data_processed.csv")
    df_product = data_cleaned[['item_id', 'name', 'rating', 'price', 'brand', 'image', 'group1']]
    user_recommendation = user_recommendation_result.merge(df_product, left_on='product_id', right_on='item_id')

    # customers list
    customers_list = user_recommendation_result[['customer_id']]
    customers_list = customers_list.drop_duplicates()

    # header
    st.markdown("<h3 style='text-align: left; color: Blue;'>2. Đề xuất dựa trên đánh giá sản phẩm</h3>", unsafe_allow_html=True)

    # choose bar
    st.markdown('***Chọn thông tin***')
    customer_id = st.selectbox('ID người dùng', customers_list['customer_id'])
    lst_num = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    items_num = st.selectbox('Số sản phẩm muốn đề xuất', lst_num)
    submit_button = st.button(label='Summit')

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
    



