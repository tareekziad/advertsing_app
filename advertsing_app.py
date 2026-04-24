
import streamlit as st
import pandas as pd 
import sklearn
import category_encoders
import joblib

st.set_page_config(layout='wide')

def get_input():
    
    Daily_Time_Spent_on_Site = st.slider('select Daily Time Spent on Site' , min_value=30 , max_value=100 ,
                                         value=45 ,step= 1)
    
    Age	= st.slider('select Age' , min_value=19 , max_value=70 ,
                                         value=45 ,step= 1)
    
    Area_Income = st.slider('select Area Income' , min_value=13996 , max_value=80484 ,
                                         value=50000 ,step= 1)
    
    Daily_Internet_Usage = st.slider('select Daily Internet Usage' , min_value=100 , max_value=270 ,
                                         value=120 ,step= 1)
    Male = st.selectbox('select gender [1 for male -- 0 for female ]' , options=[1 , 0] )
    
    Country = st.selectbox('select country ' , options = ['Tunisia', 'Nauru', 'San Marino', 'Italy', 'Iceland', 'Norway',
       'Myanmar', 'Australia', 'Grenada', 'Ghana', 'Qatar', 'Burundi',
       'Egypt', 'Bosnia and Herzegovina', 'Barbados', 'Spain',
       'Palestinian Territory', 'Afghanistan',
       'British Indian Ocean Territory (Chagos Archipelago)',
       'Russian Federation', 'Cameroon', 'Korea', 'Tokelau', 'Monaco',
       'Tuvalu', 'Greece', 'British Virgin Islands',
       'Bouvet Island (Bouvetoya)', 'Peru', 'Aruba', 'Maldives',
       'Senegal', 'Dominica', 'Luxembourg', 'Montenegro', 'Ukraine',
       'Saint Helena', 'Liberia', 'Turkmenistan', 'Niger', 'Sri Lanka',
       'Trinidad and Tobago', 'United Kingdom', 'Guinea-Bissau',
       'Micronesia', 'Turkey', 'Croatia', 'Israel',
       'Svalbard & Jan Mayen Islands', 'Azerbaijan', 'Iran',
       'Saint Vincent and the Grenadines', 'Bulgaria', 'Christmas Island',
       'Canada', 'Rwanda', 'Turks and Caicos Islands', 'Norfolk Island',
       'Cook Islands', 'Guatemala', "Cote d'Ivoire", 'Faroe Islands',
       'Ireland', 'Moldova', 'Nicaragua', 'Montserrat', 'Timor-Leste',
       'Puerto Rico', 'Central African Republic', 'Venezuela',
       'Wallis and Futuna', 'Jersey', 'Samoa',
       'Antarctica (the territory South of 60 deg S)', 'Albania',
       'Hong Kong', 'Lithuania', 'Bangladesh', 'Western Sahara', 'Serbia',
       'Czech Republic', 'Guernsey', 'Tanzania', 'Bhutan', 'Guinea',
       'Madagascar', 'Lebanon', 'Eritrea', 'Guyana',
       'United Arab Emirates', 'Martinique', 'Somalia', 'Benin',
       'Papua New Guinea', 'Uzbekistan', 'South Africa', 'Hungary',
       'Falkland Islands (Malvinas)', 'Saint Martin', 'Cuba',
       'United States Minor Outlying Islands', 'Belize', 'Kuwait',
       'Thailand', 'Gibraltar', 'Holy See (Vatican City State)',
       'Netherlands', 'Belarus', 'New Zealand', 'Togo', 'Kenya', 'Palau',
       'Cambodia', 'Costa Rica', 'Liechtenstein', 'Angola',
       'Equatorial Guinea', 'Mongolia', 'Brazil', 'Chad', 'Portugal',
       'Malawi', 'Singapore', 'Kazakhstan', 'China', 'Vietnam', 'Mayotte',
       'Jamaica', 'Bahamas', 'Algeria', 'Fiji', 'Argentina',
       'Philippines', 'Suriname', 'Guam', 'Antigua and Barbuda',
       'Georgia', 'Jordan', 'Saudi Arabia', 'Sao Tome and Principe',
       'Cyprus', 'Kyrgyz Republic', 'Pakistan', 'Seychelles',
       'Mauritania', 'Chile', 'Poland', 'Estonia', 'Latvia', 'Bahrain',
       'Colombia', 'Brunei Darussalam', 'Taiwan',
       'Saint Pierre and Miquelon', 'Finland',
       'French Southern Territories', 'Sierra Leone', 'Tajikistan',
       'Ecuador', 'Switzerland', 'France', 'Malaysia', 'Mauritius',
       'Japan', 'Greenland', 'Guadeloupe', 'Belgium', 'Honduras',
       'Paraguay', 'French Guiana', 'Northern Mariana Islands',
       'American Samoa', 'Austria', 'Tonga', 'New Caledonia',
       'United States of America', 'Morocco', 'Macedonia', 'Gabon',
       'Uganda', 'Saint Lucia', 'Niue', 'Zambia', 'Congo',
       'Pitcairn Islands', 'Anguilla', 'Sweden', 'Indonesia', 'Mexico',
       'Haiti', 'Gambia', 'El Salvador', 'Libyan Arab Jamahiriya',
       'Saint Barthelemy', 'Reunion', 'Panama', 'Dominican Republic',
       'Zimbabwe', 'Swaziland', 'Saint Kitts and Nevis', 'Burkina Faso',
       'Heard Island and McDonald Islands', 'Bolivia',
       'Netherlands Antilles', 'French Polynesia', 'Germany', 'Malta',
       'Sudan', "Lao People's Democratic Republic", 'Isle of Man',
       'Macao', 'United States Virgin Islands', 'Djibouti', 'Mali',
       'Romania', 'Cayman Islands', 'Ethiopia', 'Uruguay', 'Comoros',
       'Vanuatu', 'Nepal', 'Yemen', 'India', 'Cape Verde', 'Slovenia',
       'Denmark', 'Syrian Arab Republic', 'Andorra', 'Namibia',
       'Slovakia (Slovak Republic)', 'Armenia',
       'South Georgia and the South Sandwich Islands', 'Kiribati',
       'Marshall Islands', 'Bermuda', 'Mozambique', 'Lesotho'])

    return pd.DataFrame(data = [[Daily_Time_Spent_on_Site,Age,Area_Income,Daily_Internet_Usage,Male,Country]] ,
                 columns=['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male', 'Country'])
    

test = get_input()

lr = joblib.load('lr.pkl')

st.write(lr.predict(test))
