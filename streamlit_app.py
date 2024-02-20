import streamlit as st
from src.testdesign import design_binomial_experiment


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
    &\text{Views} \sim \text{LogNormal}(1, \sigma) \\
    &\text{CTR} \sim \text{Beta}(\alpha, \beta) \\
    &\text{Clicks} \sim \text{Binomial}(\text{Views}, \text{CTR})        
    \end{align*}
    ''')

    with st.sidebar.form(key='Data Generation Model'):
        base_ctr_pcnt = st.slider('Base CTR, %', min_value=0.1, max_value=20.0, step=0.1, value=2.0)
        uplift_pcnt = st.slider('CTR Uplift, %', min_value=0.1, max_value=40.0, step=0.1, value=1.0)
        views_sigma = st.slider('Sigma', min_value=1, max_value=500, step=1, value=200)
        ctr_beta = st.slider('Beta', min_value=1, max_value=500, step=1, value=200)
        sb_submit_button = st.form_submit_button(label='Submit')

    st.title('APP')

    st.subheader('Design Experiment Settings:')
    with st.form(key='Experiment Design'):
        col1, col2, col3 = st.columns(3)
        alpha = col1.slider(r'$\alpha$,  Type I Error:', min_value=0.01, max_value=0.2, step=0.01, value=0.05)
        beta = col2.slider(r'$\beta$, Type II Error:', min_value=0.01, max_value=0.8, step=0.01, value=0.2)
        mde = col3.slider('Minimum Detectable Effect', min_value=0.1, max_value=40.0, step=0.1, value=1.0)
        ed_submit = st.form_submit_button(label='Estimate')

    uplift = uplift_pcnt/100
    h0_ctr = base_ctr_pcnt/100
    mde = mde/100
    if ed_submit:
        n = design_binomial_experiment(min_detectable_change=mde, p_0=h0_ctr, alpha=alpha, beta=beta)
        st.write(f'p_0: {h0_ctr} uplift: {uplift} alpha: {alpha} beta: {beta} N: {n}')


if __name__ == '__main__':
    main()
