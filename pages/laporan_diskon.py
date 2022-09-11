import streamlit as st

st.markdown("""
<style>
.google-data-studio {
position: relative;
padding-bottom: 100%;
# padding-bottom:n56.25%;
padding-top: 30px; height: 0; overflow: hidden;
}

.google-data-studio iframe,
.google-data-studio object,
.google-data-studio embed {
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
}
</style>
""", unsafe_allow_html=True)


st.write('Ini adalah  salah satu report yang pernah saya buat berdasarkan  payment dan pembayaran yang saya buat menggunakan google data studio, tampilan ini di dapatkan dari query SQL dan Server lokal, untuk tampilan terbaik dapat langsung kunjungi di link berikut ini : [Link report](https://datastudio.google.com/embed/reporting/9dcdf68c-e3f8-4b57-a402-530b9298906c/page/AiB1C), dan juga anda dapat melihat jupyter notebook saya sebagai dokumentasi dari projek ini di link berikut ini, buka menggunakan google colab atau jupyter notebook :  [Link .ipynb](https://drive.google.com/file/d/1hNVl9WRlhofc71BO36Upv_GEev-wy3LP/view?usp=sharing)')
st.markdown("""
    <div class="google-data-studio">
    <iframe width="500" height="1000" src="https://datastudio.google.com/embed/reporting/9dcdf68c-e3f8-4b57-a402-530b9298906c/page/AiB1C" 
    frameborder="0" 
    style="border:0" allowfullscreen>
    </iframe></div>
    """, unsafe_allow_html=True)


