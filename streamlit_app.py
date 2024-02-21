import streamlit as st
from src.testdesign import design_binomial_experiment
from src.datagen import ABTestGenerator
import numpy as np


def main():
    st.set_page_config(
        page_title='AB-test Simulator',
        layout='centered',
        menu_items={
            'Get Help': 'https://docs.streamlit.io/',
            'About': 'AB-test Simulator app.'
            }
    )

    st.sidebar.title('Data Generation Model:')
    st.sidebar.latex(r'''
    \begin{align*}
    &\text{Views} \sim \text{LogNormal}(1, \text{skew}) \\
    &\text{CTR} \sim \text{Beta}(\alpha, \beta) \\
    &\text{Clicks} \sim \text{Binomial}(\text{Views}, \text{CTR})        
    \end{align*}
    ''')

    with st.sidebar.form(key='Data Generation Model'):
        base_ctr_pcnt = st.slider('Base CTR, %', min_value=0.1, max_value=20.0, step=0.1, value=2.0)
        uplift_pcnt = st.slider('CTR Uplift, %', min_value=0.1, max_value=40.0, step=0.1, value=1.0)
        skew = st.slider('Skew', min_value=0.1, max_value=4.0, step=0.1, value=2.0)
        ctr_beta = st.slider('Beta', min_value=1, max_value=500, step=1, value=50)
        sb_submit_button = st.form_submit_button(label='Apply')

    st.title('APP')

    st.subheader('Design Experiment Settings:')
    with st.form(key='Experiment Design'):
        col1, col2, col3 = st.columns(3)
        alpha = col1.slider(r'$\alpha$,  Type I Error:', min_value=0.01, max_value=0.2, step=0.01, value=0.05)
        n_samples = col1.slider(r'n_samples to estimate $\overline{\text{CTR}}|H_0$', min_value=100, max_value=10000, step=1, value=1000)
        beta = col2.slider(r'$\beta$, Type II Error:', min_value=0.01, max_value=0.8, step=0.01, value=0.2)
        mde = col3.slider('Minimum Detectable Effect', min_value=0.1, max_value=40.0, step=0.1, value=1.0)
        ed_submit = st.form_submit_button(label='Estimate')

    uplift = uplift_pcnt/100
    base_ctr = base_ctr_pcnt/100
    mde = mde/100

    st.write(base_ctr, uplift, ctr_beta, skew)
    datagen = ABTestGenerator(base_ctr, uplift, ctr_beta, 200.0, traffic_per_day=1000) # TODO: Think about traffic per day!
    days, ctrs_0, ctrs_1, clicks_0, clicks_1, views_0, views_1 = datagen.generate_single_experiment(n_samples)

    estimated_ctr_h0 = np.sum(clicks_0)/np.sum(views_0)
    st.write(estimated_ctr_h0, mde, alpha, skew)
    n = design_binomial_experiment(min_detectable_change=mde, p_0=estimated_ctr_h0, alpha=alpha, beta=beta)
    st.write(f'Estimated CTR: {np.round(estimated_ctr_h0, 5)} \nMinimal number of views for test: {n}')


if __name__ == '__main__':
    main()
