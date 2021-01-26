import numpy as np
import pandas as pd
from skimage.io import imread, imshow
from tqdm import tqdm
import cv2


def get_gray_hist(df):
    new_features_list = []
    for idx in tqdm(range(len(df))):
        image = imread(df.iloc[idx]['filepath'])
        image = rgb2gray(image)
        image_flat = image.flatten()
        num_bins = 20
        count, bins = np.histogram(image_flat, bins=num_bins)
        
        new_features = np.concatenate([count, bins])
        
        new_features_list.append(new_features)
    
    cols = [f'gray_hist_count_{i}' for i in range(num_bins)] + \
           [f'gray_hist_bin_edges_{i}' for i in range(num_bins+1)] 
    new_df = pd.DataFrame(new_features_list, columns=cols)
    
    return pd.concat([df, new_df], axis=1)


def rgb_stats(df_):
    
    df_rgb = df
    
    r_mean, g_mean, b_mean = [], [], []
    r_med, g_med, b_med = [], [], []
    r_std, g_std, b_std = [], [], []
    r_max, g_max, b_max = [], [], []
    
    for i in tqdm(range(len(df))):
        
        img =  imread(df['filepath'][i])
        
        #mean
        r_mean.append(np.mean(img[:,:,0]))
        g_mean.append(np.mean(img[:,:,1]))
        b_mean.append(np.mean(img[:,:,2]))
        
       #median
        r_med.append(np.median(img[:,:,0]))
        g_med.append(np.median(img[:,:,1]))
        b_med.append(np.median(img[:,:,2]))
        
        #mstd
        r_std.append(np.std(img[:,:,0]))
        g_std.append(np.std(img[:,:,1]))
        b_std.append(np.std(img[:,:,2]))
        
        #max
        r_max.append(np.max(img[:,:,0]))
        g_max.append(np.max(img[:,:,1]))
        b_max.append(np.max(img[:,:,2]))
    
    vals = [r_mean, g_mean, b_mean, r_med, g_med, b_med,
            r_std, g_std, b_std, r_max, g_max, b_max]
    cols = ['r_mean', 'g_mean', 'b_mean', 'r_med', 'g_med', 'b_med',
            'r_std', 'g_std', 'b_std', 'r_max', 'g_max', 'b_max']

    for j in vals:
        for k in cols: 
            df_rgb[k] = j
            cols.remove(k)
            break
        
    return df_rgb

def vessels_area(df):
    """Calculate the Area of the Blood Vessels via Canny Edge Detection
    """

    df_canny = df
    area = []
    for i in tqdm(range(df_canny.shape[0])):
        
        image = imread(df.loc[i,'filepath'])
        image = cv2.resize(image, (224,224))
        image = cv2.normalize(image, None, alpha = 0, beta = 255, 
                              norm_type = cv2.NORM_MINMAX, 
                              dtype =cv2.CV_8U)

        img_can = cv2.Canny(image,20,125)

        area.append(img_can.sum())
    
    df_canny['vessels_area'] = area
    
    return df_canny


def disc_area(df):
    """Calculate the Area of the Blood Vessels via Canny Edge Detection
    """

    df_canny = df
    area = []
    for i in tqdm(range(df_canny.shape[0])):
        
        image = imread(df.loc[i,'filepath'])
        image = cv2.resize(image, (224,224))
        image = cv2.normalize(image, None, alpha = 0, beta = 255, 
                                  norm_type = cv2.NORM_MINMAX, 
                                  dtype =cv2.CV_8U)
        img_can = cv2.Canny(image, 150, 165)

        img_can = dilation(erosion(erosion(dilation(dilation(
            dilation(img_can))))))

        img_can = img_can[35:189,35:189]

        cnts = cv2.findContours(img_can, cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(img_can,[c], 0, (255,), -1)

        for i in range(10):
            img_can = erosion(img_can)

        area.append(img_can.sum())
    
    df_canny['disc_area'] = area
    
    return df_canny

def hsv_stats(df_):
    
    df_rgb = df
    
    h_mean, s_mean, v_mean = [], [], []
    h_med, s_med, v_med = [], [], []
    h_std, s_std, v_std = [], [], []
    h_max, s_max, v_max = [], [], []
    
    for i in tqdm(range(len(df))):
        
        img =  imread(df['filepath'][i])
        img_hsv = rgb2hsv(img)
        
        #mean
        h_mean.append(np.mean(img_hsv[:,:,0]))
        s_mean.append(np.mean(img_hsv[:,:,1]))
        v_mean.append(np.mean(img_hsv[:,:,2]))
        
       #median
        h_med.append(np.median(img_hsv[:,:,0]))
        s_med.append(np.median(img_hsv[:,:,1]))
        v_med.append(np.median(img_hsv[:,:,2]))
        
        #mstd
        h_std.append(np.std(img_hsv[:,:,0]))
        s_std.append(np.std(img_hsv[:,:,1]))
        v_std.append(np.std(img_hsv[:,:,2]))
        
        #max
        h_max.append(np.max(img_hsv[:,:,0]))
        s_max.append(np.max(img_hsv[:,:,1]))
        v_max.append(np.max(img_hsv[:,:,2]))
    
    vals = [h_mean, s_mean, v_mean, h_med, s_med, v_med,
            h_std, s_std, v_std, h_max, s_max, v_max]
    cols = ['h_mean', 's_mean', 'v_mean', 'h_med', 's_med', 'v_med',
            'h_std', 's_std', 'v_std', 'h_max', 's_max', 'v_max']

    for j in vals:
        for k in cols: 
            df_rgb[k] = j
            cols.remove(k)
            break
        
    return df_rgb


from skimage.exposure import equalize_hist

def rgb_equalize_hist_stats(df):
    """
    Equalize the Histogram values of the RGB channels of an image and
    get
    """
    df_equalized = df
    
    r_mean, g_mean, b_mean = [], [], []
    r_med, g_med, b_med = [], [], []
    r_std, g_std, b_std = [], [], []
    r_max, g_max, b_max = [], [], []
    
    for i in range(df_equalized.shape[0]):

        image = imread(df.loc[i,'filepath'])
        image = cv2.resize(image, (224,224))
        image = cv2.normalize(image, None, alpha = 0, beta = 255, 
                                  norm_type = cv2.NORM_MINMAX, 
                                  dtype =cv2.CV_8U)
        
        r = image[:,:,0]
        g = image[:,:,1]
        b = image[:,:,2]

        r_corrected = equalize_hist(r)
        g_corrected = equalize_hist(g)
        b_corrected = equalize_hist(b)
        
        # mean
        r_mean.append(np.mean(r_corrected))
        g_mean.append(np.mean(g_corrected))
        b_mean.append(np.mean(b_corrected))
                      
        # median
        r_med.append(np.median(r_corrected))
        g_med.append(np.median(g_corrected))
        b_med.append(np.median(b_corrected))
                     
        # standard deviation
        r_std.append(np.std(r_corrected))
        g_std.append(np.std(g_corrected))
        b_std.append(np.std(b_corrected))
                     
        # max
        r_max.append(np.max(r_corrected))
        g_max.append(np.max(g_corrected))
        b_max.append(np.max(b_corrected))

    vals = [r_mean, g_mean, b_mean, r_med, g_med, b_med,
            r_std, g_std, b_std, r_max, g_max, b_max]
                     
    cols = ['r_mean_eq', 'g_mean_eq', 'b_mean_eq', 'r_med_eq', 
            'g_med_eq', 'b_med_eq',
            'r_std_eq', 'g_std_eq', 'b_std_eq', 'r_max_eq', 
            'g_max_eq', 'b_max_eq']

    for j in vals:
        for k in cols: 
            df_equalized[k] = j
            cols.remove(k)
            break
    
    return df_equalized