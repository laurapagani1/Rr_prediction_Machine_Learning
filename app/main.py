
import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def add_sidebar():
    st.sidebar.header('Ship Details')
    
    data = get_clean_data()
    
    slider_labels = [
        ('Longitudinal Center of Buoyancy', 'LC'), 
        ('Prismatic Coefficient', 'PC'),
        ('Length-to-Displacement ratio', 'L/D'),
        ('Beam-to-Draft ratio', 'B/Dr'),
        ('Length-to-Beam ratio', 'L/B'),
        ('Froude number', 'Fr')
    ]
    
    input_dict = {}
    
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value = float(data[key].min()),
            max_value = float (data[key].max()), 
            value =   float (data[key].mean())
        )
    
    return input_dict
        
def get_clean_data():
    
    data =pd.read_csv('data/yacht_hydro.csv')
    print(data.head())
    print(data.describe())
    return data

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['Rr'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
  
    categories = ['Longitudinal Center of Buoyancy', 'Prismatic Coefficient', 'Length-to-Displacement ratio', 'Beam-to-Draft ratio', 
                    'Length-to-Beam ratio', 'Froude number']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['LC'], input_data['PC'], input_data['L/D'],
            input_data['B/Dr'], input_data['L/B'], input_data['Fr']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig
    

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    #st.write(input_array)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Ship resistance")
    st.write("The Rr for the values selected is:")
    
    st.write(prediction[0])
    #st.write(prediction)
    #st.write("Prediction of Rr: ", model.predict_proba(input_array_scaled)[0][0])

    st.write("This app can assist you in calculating Rr")

    
    
def main():
    st.set_page_config(
        page_title = "Machine Learning Rr app Calculator",
        page_icon=':smile:',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()
    #st.write(input_data)
    
    
    with st.container():
        st.title('Machine Learning Rr app Calculator')
        st.write('This is a simple machine learning app to retrieve Rr, which is the resistance encountered by the ship as it moves through the water. This is the target variable in the dataset and can include various forms of resistance such as frictional, wave-making, and viscous resistance.')
    
    col1, col2 = st.columns([4,1])
    
    with col1: 
        #st.write('This in column 1')
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
        st.subheader('Ship Features Explanation')
        st.write('<ul>', unsafe_allow_html=True)
        st.write('<li>LC: The position of the center of buoyancy along the length of the ship. It is typically measured from a reference point, such as the bow (front) of the ship. A negative value often indicates that the center of buoyancy is aft (towards the rear) of the midpoint.</li>', unsafe_allow_html=True)
        st.write('<li>PC: A dimensionless coefficient that compares the volume of a ship\'s hull to the volume of a prism with the same length and maximum cross-sectional area. It gives an indication of the fullness of the hull form. Higher values suggest a fuller hull, which typically has lower resistance at lower speeds.</li>', unsafe_allow_html=True)
        st.write('<li>L/D: The ratio of the length of the ship to its displacement. This ratio gives an indication of the slenderness of the hull. A higher ratio generally means a more slender hull, which tends to have lower resistance at higher speeds</li>', unsafe_allow_html=True)
        st.write('<li>B/Dr: The ratio of the beam (width) of the ship to its draft (the vertical distance between the waterline and the bottom of the hull). This ratio affects the stability and resistance of the ship. A higher ratio usually implies a wider and shallower hull, which can affect resistance and stability.</li>', unsafe_allow_html=True)
        st.write('<li>L/B: The ratio of the length of the ship to its beam (width). This ratio provides an indication of the slenderness of the hull. Higher values suggest a more slender hull, which typically reduces resistance, especially at higher speeds.</li>', unsafe_allow_html=True)
        st.write('<li>Fr: The Froude number is crucial in naval architecture as it influences the wave-making resistance of the ship</li>', unsafe_allow_html=True)
        st.write('</ul>', unsafe_allow_html=True)

    with col2: 
        #st.write('This in column 2')
        add_predictions(input_data)
        
   # st.image('img/logo.png', caption='', use_column_width=True)


if __name__ == '__main__':
    main()
