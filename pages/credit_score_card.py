import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import numpy as np

st.write("""
# App Score Card - Resiko Kredit Berdasarkan Waktu Pembayaran Pinjaman
Aplikasi ini memprediksi **Resiko Kredit yang dimiliki pelanggan berdasarkan prilaku dalam membayaran pinjaman**!, jika telat bayar lebih dari 30 hari maka akan di klasifikasikan sebagai peminjam yang buruk, namun jika di bawah ini maka akan diklasifikasikan sebagai peminjam yang baik, 
[Link dokumentasi Machine Learning di Github](https://github.com/irvansikajudin/Credit-Risk-with-AUC-and-Kolmogorov-Smirnov/). 
""")


# Sidebar
# Header of Specify Input Parameters
st.sidebar.info('### Pilih Parameter Input')

tipekarakteristikbad = st.sidebar.selectbox('Mau liat karakteristik bad loan?',('Tidak','Ya'))

def user_input_features():
    if tipekarakteristikbad == 'Tidak':
        grade_B=st.sidebar.slider('grade_B',0,1,1)
        grade_C=st.sidebar.slider('grade_C',0,1,0)
        grade_D=st.sidebar.slider('grade_D',0,1,0)
        grade_E=st.sidebar.slider('grade_E',0,1,0)
        grade_F=st.sidebar.slider('grade_F',0,1,0)
        grade_G=st.sidebar.slider('grade_G',0,1,0)
        home_ownership_MORTGAGE=st.sidebar.slider('home_ownership_MORTGAGE',0,1,0)
        home_ownership_NONE=st.sidebar.slider('home_ownership_NONE',0,1,0)
        home_ownership_OTHER=st.sidebar.slider('home_ownership_OTHER',0,1,0)
        home_ownership_OWN=st.sidebar.slider('home_ownership_OWN',0,1,0)
        home_ownership_RENT=st.sidebar.slider('home_ownership_RENT',0,1,1)
        verification_status_Source_Verified=st.sidebar.slider('verification_status_Source_Verified',0,1,0)
        verification_status_Verified=st.sidebar.slider('verification_status_Verified',0,1,1)
        pymnt_plan_y=st.sidebar.slider('pymnt_plan_y',0,1,0)
        purpose_credit_card=st.sidebar.slider('purpose_credit_card',0,1,1)
        purpose_debt_consolidation=st.sidebar.slider('purpose_debt_consolidation',0,1,0)
        purpose_educational=st.sidebar.slider('purpose_educational',0,1,0)
        purpose_home_improvement=st.sidebar.slider('purpose_home_improvement',0,1,0)
        purpose_house=st.sidebar.slider('purpose_house',0,1,0)
        purpose_major_purchase=st.sidebar.slider('purpose_major_purchase',0,1,0)
        purpose_medical=st.sidebar.slider('purpose_medical',0,1,0)
        purpose_moving=st.sidebar.slider('purpose_moving',0,1,0)
        purpose_other=st.sidebar.slider('purpose_other',0,1,0)
        purpose_renewable_energy=st.sidebar.slider('purpose_renewable_energy',0,1,0)
        purpose_small_business=st.sidebar.slider('purpose_small_business',0,1,0)
        purpose_vacation=st.sidebar.slider('purpose_vacation',0,1,0)
        purpose_wedding=st.sidebar.slider('purpose_wedding',0,1,0)
        addr_state_AL=st.sidebar.slider('addr_state_AL',0,1,0)
        addr_state_AR=st.sidebar.slider('addr_state_AR',0,1,0)
        addr_state_AZ=st.sidebar.slider('addr_state_AZ',0,1,1)
        addr_state_CA=st.sidebar.slider('addr_state_CA',0,1,0)
        addr_state_CO=st.sidebar.slider('addr_state_CO',0,1,0)
        addr_state_CT=st.sidebar.slider('addr_state_CT',0,1,0)
        addr_state_DC=st.sidebar.slider('addr_state_DC',0,1,0)
        addr_state_DE=st.sidebar.slider('addr_state_DE',0,1,0)
        addr_state_FL=st.sidebar.slider('addr_state_FL',0,1,0)
        addr_state_GA=st.sidebar.slider('addr_state_GA',0,1,0)
        addr_state_HI=st.sidebar.slider('addr_state_HI',0,1,0)
        addr_state_IA=st.sidebar.slider('addr_state_IA',0,1,0)
        addr_state_ID=st.sidebar.slider('addr_state_ID',0,1,0)
        addr_state_IL=st.sidebar.slider('addr_state_IL',0,1,0)
        addr_state_IN=st.sidebar.slider('addr_state_IN',0,1,0)
        addr_state_KS=st.sidebar.slider('addr_state_KS',0,1,0)
        addr_state_KY=st.sidebar.slider('addr_state_KY',0,1,0)
        addr_state_LA=st.sidebar.slider('addr_state_LA',0,1,0)
        addr_state_MA=st.sidebar.slider('addr_state_MA',0,1,0)
        addr_state_MD=st.sidebar.slider('addr_state_MD',0,1,0)
        addr_state_ME=st.sidebar.slider('addr_state_ME',0,1,0)
        addr_state_MI=st.sidebar.slider('addr_state_MI',0,1,0)
        addr_state_MN=st.sidebar.slider('addr_state_MN',0,1,0)
        addr_state_MO=st.sidebar.slider('addr_state_MO',0,1,0)
        addr_state_MS=st.sidebar.slider('addr_state_MS',0,1,0)
        addr_state_MT=st.sidebar.slider('addr_state_MT',0,1,0)
        addr_state_NC=st.sidebar.slider('addr_state_NC',0,1,0)
        addr_state_NE=st.sidebar.slider('addr_state_NE',0,1,0)
        addr_state_NH=st.sidebar.slider('addr_state_NH',0,1,0)
        addr_state_NJ=st.sidebar.slider('addr_state_NJ',0,1,0)
        addr_state_NM=st.sidebar.slider('addr_state_NM',0,1,0)
        addr_state_NV=st.sidebar.slider('addr_state_NV',0,1,0)
        addr_state_NY=st.sidebar.slider('addr_state_NY',0,1,0)
        addr_state_OH=st.sidebar.slider('addr_state_OH',0,1,0)
        addr_state_OK=st.sidebar.slider('addr_state_OK',0,1,0)
        addr_state_OR=st.sidebar.slider('addr_state_OR',0,1,0)
        addr_state_PA=st.sidebar.slider('addr_state_PA',0,1,0)
        addr_state_RI=st.sidebar.slider('addr_state_RI',0,1,0)
        addr_state_SC=st.sidebar.slider('addr_state_SC',0,1,0)
        addr_state_SD=st.sidebar.slider('addr_state_SD',0,1,0)
        addr_state_TN=st.sidebar.slider('addr_state_TN',0,1,0)
        addr_state_TX=st.sidebar.slider('addr_state_TX',0,1,0)
        addr_state_UT=st.sidebar.slider('addr_state_UT',0,1,0)
        addr_state_VA=st.sidebar.slider('addr_state_VA',0,1,0)
        addr_state_VT=st.sidebar.slider('addr_state_VT',0,1,0)
        addr_state_WA=st.sidebar.slider('addr_state_WA',0,1,0)
        addr_state_WI=st.sidebar.slider('addr_state_WI',0,1,0)
        addr_state_WV=st.sidebar.slider('addr_state_WV',0,1,0)
        addr_state_WY=st.sidebar.slider('addr_state_WY',0,1,0)
        initial_list_status_w=st.sidebar.slider('initial_list_status_w',0,1,0)
        Unnamed_0=st.sidebar.slider('Unnamed_0',-1.726925292795039,1.737353858733435,-1.726925292795039)
        loan_amnt=st.sidebar.slider('loan_amnt',-1.681679717527286,2.476518426110183,-1.139306046618051)
        int_rate=st.sidebar.slider('int_rate',-1.9269822181280545,2.8065177634845844,-0.7275536859655491)
        annual_inc=st.sidebar.slider('annual_inc',-1.3047465932369473,133.63868680987443,-0.9069407409918498)
        dti=st.sidebar.slider('dti',-2.195046864867256,2.9185506473224323,1.340611329982578)
        delinq_2yrs=st.sidebar.slider('delinq_2yrs',-0.3582082456691558,35.91772728739664,-0.3582082456691558)
        inq_last_6mths=st.sidebar.slider('inq_last_6mths',-0.7383436406043477,29.47347464800061,0.1771660045048934)
        mths_since_last_delinq=st.sidebar.slider('mths_since_last_delinq',-0.7082022633374644,7.54676799099713,-0.7082022633374644)
        open_acc=st.sidebar.slider('open_acc',-2.050639658933913,14.558678169294929,-1.6504151329524954)
        pub_rec=st.sidebar.slider('pub_rec',-0.3061969848422947,124.536588245482,-0.3061969848422947)
        revol_bal=st.sidebar.slider('revol_bal',-0.7850849268540918,122.4137418148634,-0.130580894198925)
        revol_util=st.sidebar.slider('revol_util',-2.3725033758769447,35.23798437854584,1.1554556345962004)
        total_acc=st.sidebar.slider('total_acc',-2.079947308793255,11.283127823540926,-1.3902402051889104)
        out_prncp=st.sidebar.slider('out_prncp',-0.6934060959967274,4.33522021494412,-0.6934060959967274)
        total_rec_late_fee=st.sidebar.slider('total_rec_late_fee',-0.1239159555071231,67.62972450158458,-0.1239159555071231)
        recoveries=st.sidebar.slider('recoveries',-0.1550738332097679,60.07499338348107,-0.1550738332097679)
        collections_12_mths_ex_med=st.sidebar.slider('collections_12_mths_ex_med',-0.0831689316074092,184.6685657026672,-0.0831689316074092)
        acc_now_delinq=st.sidebar.slider('acc_now_delinq',-0.0583211191979827,72.61208050420211,-0.0583211191979827)
        tot_coll_amt=st.sidebar.slider('tot_coll_amt',-0.011605264654138,663.3298073631237,-0.011605264654138)
        tot_cur_bal=st.sidebar.slider('tot_cur_bal',-0.7970641290136494,52.55343203141644,-0.7970641290136494)
        emp_length_int=st.sidebar.slider('emp_length_int',-1.6530098776791071,1.1044509887535456,1.1044509887535456)
        term_int=st.sidebar.slider('term_int',-0.6232129346997329,1.6045880056738633,-0.6232129346997329)
        bulan_berlalu_since_earliest_cr_line=st.sidebar.slider('bulan_berlalu_since_earliest_cr_line',-1.984504468560013,7.703691802593736,1.8479269802256224)
        bulan_berlalu_since_issue_d=st.sidebar.slider('bulan_berlalu_since_issue_d',-1.0665968923025166,5.046440592350572,1.4341911696010197)
    else:
        grade_B=st.sidebar.slider('grade_B',0,1,0)
        grade_C=st.sidebar.slider('grade_C',0,1,1)
        grade_D=st.sidebar.slider('grade_D',0,1,0)
        grade_E=st.sidebar.slider('grade_E',0,1,0)
        grade_F=st.sidebar.slider('grade_F',0,1,0)
        grade_G=st.sidebar.slider('grade_G',0,1,0)
        home_ownership_MORTGAGE=st.sidebar.slider('home_ownership_MORTGAGE',0,1,0)
        home_ownership_NONE=st.sidebar.slider('home_ownership_NONE',0,1,0)
        home_ownership_OTHER=st.sidebar.slider('home_ownership_OTHER',0,1,0)
        home_ownership_OWN=st.sidebar.slider('home_ownership_OWN',0,1,0)
        home_ownership_RENT=st.sidebar.slider('home_ownership_RENT',0,1,1)
        verification_status_Source_Verified=st.sidebar.slider('verification_status_Source_Verified',0,1,1)
        verification_status_Verified=st.sidebar.slider('verification_status_Verified',0,1,0)
        pymnt_plan_y=st.sidebar.slider('pymnt_plan_y',0,1,0)
        purpose_credit_card=st.sidebar.slider('purpose_credit_card',0,1,0)
        purpose_debt_consolidation=st.sidebar.slider('purpose_debt_consolidation',0,1,0)
        purpose_educational=st.sidebar.slider('purpose_educational',0,1,0)
        purpose_home_improvement=st.sidebar.slider('purpose_home_improvement',0,1,0)
        purpose_house=st.sidebar.slider('purpose_house',0,1,0)
        purpose_major_purchase=st.sidebar.slider('purpose_major_purchase',0,1,0)
        purpose_medical=st.sidebar.slider('purpose_medical',0,1,0)
        purpose_moving=st.sidebar.slider('purpose_moving',0,1,0)
        purpose_other=st.sidebar.slider('purpose_other',0,1,0)
        purpose_renewable_energy=st.sidebar.slider('purpose_renewable_energy',0,1,0)
        purpose_small_business=st.sidebar.slider('purpose_small_business',0,1,0)
        purpose_vacation=st.sidebar.slider('purpose_vacation',0,1,0)
        purpose_wedding=st.sidebar.slider('purpose_wedding',0,1,0)
        addr_state_AL=st.sidebar.slider('addr_state_AL',0,1,0)
        addr_state_AR=st.sidebar.slider('addr_state_AR',0,1,0)
        addr_state_AZ=st.sidebar.slider('addr_state_AZ',0,1,0)
        addr_state_CA=st.sidebar.slider('addr_state_CA',0,1,0)
        addr_state_CO=st.sidebar.slider('addr_state_CO',0,1,0)
        addr_state_CT=st.sidebar.slider('addr_state_CT',0,1,0)
        addr_state_DC=st.sidebar.slider('addr_state_DC',0,1,0)
        addr_state_DE=st.sidebar.slider('addr_state_DE',0,1,0)
        addr_state_FL=st.sidebar.slider('addr_state_FL',0,1,0)
        addr_state_GA=st.sidebar.slider('addr_state_GA',0,1,1)
        addr_state_HI=st.sidebar.slider('addr_state_HI',0,1,0)
        addr_state_IA=st.sidebar.slider('addr_state_IA',0,1,0)
        addr_state_ID=st.sidebar.slider('addr_state_ID',0,1,0)
        addr_state_IL=st.sidebar.slider('addr_state_IL',0,1,0)
        addr_state_IN=st.sidebar.slider('addr_state_IN',0,1,0)
        addr_state_KS=st.sidebar.slider('addr_state_KS',0,1,0)
        addr_state_KY=st.sidebar.slider('addr_state_KY',0,1,0)
        addr_state_LA=st.sidebar.slider('addr_state_LA',0,1,0)
        addr_state_MA=st.sidebar.slider('addr_state_MA',0,1,0)
        addr_state_MD=st.sidebar.slider('addr_state_MD',0,1,0)
        addr_state_ME=st.sidebar.slider('addr_state_ME',0,1,0)
        addr_state_MI=st.sidebar.slider('addr_state_MI',0,1,0)
        addr_state_MN=st.sidebar.slider('addr_state_MN',0,1,0)
        addr_state_MO=st.sidebar.slider('addr_state_MO',0,1,0)
        addr_state_MS=st.sidebar.slider('addr_state_MS',0,1,0)
        addr_state_MT=st.sidebar.slider('addr_state_MT',0,1,0)
        addr_state_NC=st.sidebar.slider('addr_state_NC',0,1,0)
        addr_state_NE=st.sidebar.slider('addr_state_NE',0,1,0)
        addr_state_NH=st.sidebar.slider('addr_state_NH',0,1,0)
        addr_state_NJ=st.sidebar.slider('addr_state_NJ',0,1,0)
        addr_state_NM=st.sidebar.slider('addr_state_NM',0,1,0)
        addr_state_NV=st.sidebar.slider('addr_state_NV',0,1,0)
        addr_state_NY=st.sidebar.slider('addr_state_NY',0,1,0)
        addr_state_OH=st.sidebar.slider('addr_state_OH',0,1,0)
        addr_state_OK=st.sidebar.slider('addr_state_OK',0,1,0)
        addr_state_OR=st.sidebar.slider('addr_state_OR',0,1,0)
        addr_state_PA=st.sidebar.slider('addr_state_PA',0,1,0)
        addr_state_RI=st.sidebar.slider('addr_state_RI',0,1,0)
        addr_state_SC=st.sidebar.slider('addr_state_SC',0,1,0)
        addr_state_SD=st.sidebar.slider('addr_state_SD',0,1,0)
        addr_state_TN=st.sidebar.slider('addr_state_TN',0,1,0)
        addr_state_TX=st.sidebar.slider('addr_state_TX',0,1,0)
        addr_state_UT=st.sidebar.slider('addr_state_UT',0,1,0)
        addr_state_VA=st.sidebar.slider('addr_state_VA',0,1,0)
        addr_state_VT=st.sidebar.slider('addr_state_VT',0,1,0)
        addr_state_WA=st.sidebar.slider('addr_state_WA',0,1,0)
        addr_state_WI=st.sidebar.slider('addr_state_WI',0,1,0)
        addr_state_WV=st.sidebar.slider('addr_state_WV',0,1,0)
        addr_state_WY=st.sidebar.slider('addr_state_WY',0,1,0)
        initial_list_status_w=st.sidebar.slider('initial_list_status_w',0,1,0)
        Unnamed_0=st.sidebar.slider('Unnamed_0',-1.726925292795039,1.737353858733435,-1.7269178632474853)
        loan_amnt=st.sidebar.slider('loan_amnt',-1.681679717527286,2.476518426110183,-1.440624752678737)
        int_rate=st.sidebar.slider('int_rate',-1.9269822181280545,2.8065177634845844,0.3319797401512332)
        annual_inc=st.sidebar.slider('annual_inc',-1.3047465932369473,133.63868680987443,-0.79895869640849)
        dti=st.sidebar.slider('dti',-2.195046864867256,2.9185506473224323,-2.067174959086068)
        delinq_2yrs=st.sidebar.slider('delinq_2yrs',-0.3582082456691558,35.91772728739664,-0.3582082456691558)
        inq_last_6mths=st.sidebar.slider('inq_last_6mths',-0.7383436406043477,29.47347464800061,3.839204584941858)
        mths_since_last_delinq=st.sidebar.slider('mths_since_last_delinq',-0.7082022633374644,7.54676799099713,-0.7082022633374644)
        open_acc=st.sidebar.slider('open_acc',-2.050639658933913,14.558678169294929,-1.6504151329524954)
        pub_rec=st.sidebar.slider('pub_rec',-0.3061969848422947,124.536588245482,-0.3061969848422947)
        revol_bal=st.sidebar.slider('revol_bal',-0.7850849268540918,122.4137418148634,-0.7041830875304351)
        revol_util=st.sidebar.slider('revol_util',-2.3725033758769447,35.23798437854584,-1.9762929254773325)
        total_acc=st.sidebar.slider('total_acc',-2.079947308793255,11.283127823540926,-1.8213071449416256)
        out_prncp=st.sidebar.slider('out_prncp',-0.6934060959967274,4.33522021494412,-0.6934060959967274)
        total_rec_late_fee=st.sidebar.slider('total_rec_late_fee',-0.1239159555071231,67.62972450158458,-0.1239159555071231)
        recoveries=st.sidebar.slider('recoveries',-0.1550738332097679,60.07499338348107,0.0552984659909892)
        collections_12_mths_ex_med=st.sidebar.slider('collections_12_mths_ex_med',-0.0831689316074092,184.6685657026672,-0.0831689316074092)
        acc_now_delinq=st.sidebar.slider('acc_now_delinq',-0.0583211191979827,72.61208050420211,-0.0583211191979827)
        tot_coll_amt=st.sidebar.slider('tot_coll_amt',-0.011605264654138,663.3298073631237,-0.011605264654138)
        tot_cur_bal=st.sidebar.slider('tot_cur_bal',-0.7970641290136494,52.55343203141644,-0.7970641290136494)
        emp_length_int=st.sidebar.slider('emp_length_int',-1.6530098776791071,1.1044509887535456,-1.6530098776791071)
        term_int=st.sidebar.slider('term_int',-0.6232129346997329,1.6045880056738633,1.6045880056738633)
        bulan_berlalu_since_earliest_cr_line=st.sidebar.slider('bulan_berlalu_since_earliest_cr_line',-1.984504468560013,7.703691802593736,-0.1873083543779293)
        bulan_berlalu_since_issue_d=st.sidebar.slider('bulan_berlalu_since_issue_d',-1.0665968923025166,5.046440592350572,1.4341911696010197)
    data = {
            'grade_B':grade_B,
            'grade_C':grade_C,
            'grade_D':grade_D,
            'grade_E':grade_E,
            'grade_F':grade_F,
            'grade_G':grade_G,
            'home_ownership_MORTGAGE':home_ownership_MORTGAGE,
            'home_ownership_NONE':home_ownership_NONE,
            'home_ownership_OTHER':home_ownership_OTHER,
            'home_ownership_OWN':home_ownership_OWN,
            'home_ownership_RENT':home_ownership_RENT,
            'verification_status_Source_Verified':verification_status_Source_Verified,
            'verification_status_Verified':verification_status_Verified,
            'pymnt_plan_y':pymnt_plan_y,
            'purpose_credit_card':purpose_credit_card,
            'purpose_debt_consolidation':purpose_debt_consolidation,
            'purpose_educational':purpose_educational,
            'purpose_home_improvement':purpose_home_improvement,
            'purpose_house':purpose_house,
            'purpose_major_purchase':purpose_major_purchase,
            'purpose_medical':purpose_medical,
            'purpose_moving':purpose_moving,
            'purpose_other':purpose_other,
            'purpose_renewable_energy':purpose_renewable_energy,
            'purpose_small_business':purpose_small_business,
            'purpose_vacation':purpose_vacation,
            'purpose_wedding':purpose_wedding,
            'addr_state_AL':addr_state_AL,
            'addr_state_AR':addr_state_AR,
            'addr_state_AZ':addr_state_AZ,
            'addr_state_CA':addr_state_CA,
            'addr_state_CO':addr_state_CO,
            'addr_state_CT':addr_state_CT,
            'addr_state_DC':addr_state_DC,
            'addr_state_DE':addr_state_DE,
            'addr_state_FL':addr_state_FL,
            'addr_state_GA':addr_state_GA,
            'addr_state_HI':addr_state_HI,
            'addr_state_IA':addr_state_IA,
            'addr_state_ID':addr_state_ID,
            'addr_state_IL':addr_state_IL,
            'addr_state_IN':addr_state_IN,
            'addr_state_KS':addr_state_KS,
            'addr_state_KY':addr_state_KY,
            'addr_state_LA':addr_state_LA,
            'addr_state_MA':addr_state_MA,
            'addr_state_MD':addr_state_MD,
            'addr_state_ME':addr_state_ME,
            'addr_state_MI':addr_state_MI,
            'addr_state_MN':addr_state_MN,
            'addr_state_MO':addr_state_MO,
            'addr_state_MS':addr_state_MS,
            'addr_state_MT':addr_state_MT,
            'addr_state_NC':addr_state_NC,
            'addr_state_NE':addr_state_NE,
            'addr_state_NH':addr_state_NH,
            'addr_state_NJ':addr_state_NJ,
            'addr_state_NM':addr_state_NM,
            'addr_state_NV':addr_state_NV,
            'addr_state_NY':addr_state_NY,
            'addr_state_OH':addr_state_OH,
            'addr_state_OK':addr_state_OK,
            'addr_state_OR':addr_state_OR,
            'addr_state_PA':addr_state_PA,
            'addr_state_RI':addr_state_RI,
            'addr_state_SC':addr_state_SC,
            'addr_state_SD':addr_state_SD,
            'addr_state_TN':addr_state_TN,
            'addr_state_TX':addr_state_TX,
            'addr_state_UT':addr_state_UT,
            'addr_state_VA':addr_state_VA,
            'addr_state_VT':addr_state_VT,
            'addr_state_WA':addr_state_WA,
            'addr_state_WI':addr_state_WI,
            'addr_state_WV':addr_state_WV,
            'addr_state_WY':addr_state_WY,
            'initial_list_status_w':initial_list_status_w,
            'Unnamed_0':Unnamed_0,
            'loan_amnt':loan_amnt,
            'int_rate':int_rate,
            'annual_inc':annual_inc,
            'dti':dti,
            'delinq_2yrs':delinq_2yrs,
            'inq_last_6mths':inq_last_6mths,
            'mths_since_last_delinq':mths_since_last_delinq,
            'open_acc':open_acc,
            'pub_rec':pub_rec,
            'revol_bal':revol_bal,
            'revol_util':revol_util,
            'total_acc':total_acc,
            'out_prncp':out_prncp,
            'total_rec_late_fee':total_rec_late_fee,
            'recoveries':recoveries,
            'collections_12_mths_ex_med':collections_12_mths_ex_med,
            'acc_now_delinq':acc_now_delinq,
            'tot_coll_amt':tot_coll_amt,
            'tot_cur_bal':tot_cur_bal,
            'emp_length_int':emp_length_int,
            'term_int':term_int,
            'bulan_berlalu_since_earliest_cr_line':bulan_berlalu_since_earliest_cr_line,
            'bulan_berlalu_since_issue_d':bulan_berlalu_since_issue_d
            }
    features = pd.DataFrame(data, index=[0])
    return features


# feature_imp = st.sidebar.selectbox('Mau pake features important?',('Tidak','Ya'))
# st.write('You selected:', feature_imp)

# if feature_imp == 'Tidak':
#     st.warning('Kamu lagi ga pake fitur important ya..!, kalo kamu mau pake, aktifin di sidebar ya :), tapi kalo kamu pake, Machine Learning akan lebih berat sehingga loadingnya bakal lebih lama')
# else:
#     st.info('Kamu lagi pake Features Important, ingat Machine learning jadi lebih berat, loading bakal lebih lama ya!.')
# st.write('---')

df = user_input_features() 

# Main Panel


# load the model from disk
from joblib import dump, load
model = load(open('ml_model/rfcmodel_credit_risk_auc_kolmogorov.pkl', 'rb'))
prediction = model.predict(df)
st.header('Perkiraan Resiko Pelanggan')

if prediction.item(0) == 1:
    st.error('###### Pelanggan dengan karakteristik yang telah dipilih diperkirakan memiliki prilaku yang buruk dalam membayar pinjaman, lebih dari 30 hari.')
else:
    st.info('###### Pelanggan dengan karakteristik yang telah dipilih diperkirakan memiliki prilaku yang baik dalam membayar pinjaman.')
st.write('---')


# Print specified input parameters
st.header('Parameter input dipilih')
st.write(df)
st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# # hanya gunakanan X_test untuk melihat feature important

# if feature_imp == 'Ya':
#     explainer = shap.TreeExplainer(model)
#     # shap_values = explainer.shap_values(X)
#     shap_values = explainer.shap_values(X_test)
#     st.set_option('deprecation.showPyplotGlobalUse', False) #untuk memastikan fitur berfungsi baik di versi terbaru
#     st.header('Feature Importance')
#     plt.title('Feature importance based on SHAP values')
#     # shap.summary_plot(shap_values, X)
#     shap.summary_plot(shap_values, X_test)
#     st.pyplot(bbox_inches='tight')
#     st.write('---')
#     plt.title('Feature importance based on SHAP values (Bar)')
#     shap.summary_plot(shap_values, X, plot_type="bar")
#     st.pyplot(bbox_inches='tight')
# else:
#     st.warning('---')


