import streamlit as st
from src.testdesign import design_binomial_experiment
from src.datagen import ABTestGenerator
from src.plots import plot_ctr, plot_views, plot_p_hist, plot_power, plot_p_cdf
from src.utils import apply_tests
from src.tests import t_test, mw_test
import numpy as np


def hash_func(x: ABTestGenerator):
    return hash((x.base_ctr, x.uplift, x.beta, x.skew, x.traffic_per_day))


@st.cache_data(hash_funcs={ABTestGenerator: hash_func})
def generate_ab_test_data(datagenerator: ABTestGenerator, num_users, n_runs):
    return datagenerator.generate_n_experiment(num_users, n_runs)


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
        uplift_pcnt = st.slider('CTR Uplift, %', min_value=0.1, max_value=10.0, step=0.1, value=0.5)
        skew = st.slider('Skew', min_value=0.1, max_value=4.0, step=0.1, value=0.6)
        ctr_beta = st.slider('Beta', min_value=1, max_value=500, step=1, value=250)
        sb_submit_button = st.form_submit_button(label='Apply')

    st.title('APP')

    st.subheader('Design Experiment Settings:')
    with st.form(key='Experiment Design'):
        col1, col2, col3 = st.columns(3)
        alpha = col1.slider(r'$\alpha$,  Type I Error:', min_value=0.01, max_value=0.2, step=0.01, value=0.05)
        n_samples = col1.slider(r'n_samples to estimate $\overline{\text{CTR}}|H_0$', min_value=100, max_value=10000, step=1, value=1000)
        beta = col2.slider(r'$\beta$, Type II Error:', min_value=0.01, max_value=0.8, step=0.01, value=0.2)
        mde = col3.slider('Minimum Detectable Effect', min_value=0.1, max_value=10.0, step=0.1, value=0.5)
        ed_submit = st.form_submit_button(label='Estimate')

    uplift = uplift_pcnt/100
    base_ctr = base_ctr_pcnt/100
    mde = mde/100

    st.write(f'base_ctr: {base_ctr} uplift: {uplift}ctr_beta: {ctr_beta} skew: {skew}')

    datagen_aa = ABTestGenerator(base_ctr, 0, ctr_beta, skew, traffic_per_day=1000) # TODO: Think about traffic per day!

    result_dict_estimation = generate_ab_test_data(datagen_aa, n_samples, 1) # TODO: handle n_runs
    clicks_0 = result_dict_estimation['clicks_0'][0]
    st.write(f"len clicks: {result_dict_estimation['clicks_0'].shape}")
    st.write(f'len clicks[0]: {clicks_0.shape}')
    views_0 = result_dict_estimation['views_0'][0]
    estimated_ctr_h0 = np.sum(clicks_0)/np.sum(views_0)
    st.write(f'base_ctr: {base_ctr} estimated_ctr_h0: {estimated_ctr_h0} mde: {mde} alpha: {alpha}, skew: {skew}')

    min_samples_required = design_binomial_experiment(min_detectable_change=mde, p_0=estimated_ctr_h0, alpha=alpha, beta=beta)
    st.write(f'Estimated CTR: {np.round(estimated_ctr_h0, 5)} \nMinimal number of views for test: {min_samples_required}')

    n_samples = 4000
    datagen_ab = ABTestGenerator(base_ctr, uplift, ctr_beta, skew, traffic_per_day=1000)
    result_dict_aa = generate_ab_test_data(datagen_aa, n_samples, 1000)
    result_dict_ab = generate_ab_test_data(datagen_ab, n_samples, 1000)


    c1, c2 = st.columns([1, 1])
    st.subheader("Ground Truth Distributions under H0 and H1.")
    with c1:
        st.write('A/A')
        plot_ctr(result_dict_aa, 0)
        plot_views(result_dict_aa, 0)
    with c2:
        st.write('A/B')
        plot_ctr(result_dict_ab, 0)
        plot_views(result_dict_ab, 0)

    # A/B testing part
    test_config={'t_test': t_test, 'mw_test': mw_test}
    st.subheader('A/B tests results.')
    p_vals_aa = apply_tests(result_dict_aa, test_config=test_config)
    p_vals_ab = apply_tests(result_dict_ab, test_config=test_config)

    plot_p_hist(p_vals_aa['t_test']['p_vals'])
    #plot_p_hist2(p_vals)
    plot_p_cdf(p_vals_ab['t_test']['p_vals'])
    plot_power(p_vals_ab, alpha=0.05)

if __name__ == '__main__':
    main()
