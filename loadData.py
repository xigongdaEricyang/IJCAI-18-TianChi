import pandas as pd
import time

path = '../RawData/round1_ijcai_18_train_20180301.txt'
raw_df = pd.read_table(path, header=0, sep=' ').drop_duplicates().drop([
            'item_id', 'user_id', 'context_id','instance_id', 'shop_id'], axis=1)
raw_df['context_time'] = raw_df.apply(lambda item: time.ctime(item.context_timestamp), axis=1)

ad_info_columns = ['item_category_list', 'item_property_list', 'item_brand_id', 
                   'item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level']
user_info_columns = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
context_info_columns = ['context_timestamp', 'context_page_id', 'predict_category_property']
shop_info_columns = ['shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 
                     'shop_score_service', 'shop_score_delivery', 'shop_score_description']
result_columns = ['is_trade']
ad_info_df = raw_df[ad_info_columns]
user_info_df = raw_df[user_info_columns]
context_info_df = raw_df[context_info_columns]
shop_info_df = raw_df[shop_info_columns]
result_df = raw_df[result_columns]